import cv2
import numpy as np
import logging
import json
import os

from .motion_gating import MotionGating
from .detector import YOLODetector
from .tracking import DeepSORTTracker
from .zones import ZoneAnalyzer
from .event_detection import EventGenerator
from .pose_estimator import PoseEstimator

log = logging.getLogger("cv_pipeline")

class CVPipeline:
    def __init__(self):
        """Initialize CV pipeline components."""
        self.motion = MotionGating(threshold=25)
        self.detector = YOLODetector()
        self.tracker = DeepSORTTracker()
        self.zone_analyzer = ZoneAnalyzer()
        self.event_gen = EventGenerator()
        self.pose_estimator = PoseEstimator()
        log.info("[CV] ✅ Pipeline initialized (with MediaPipe Pose)")

    def process_video(self, video_path: str, output_dir="json_outputs"):
        """
        Robust video processing with error handling.
        Returns: (events_list, salient_frames_list)
        """
        log.info(f"[CV] Processing: {video_path}")
        
        # Extract video name
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        log.info(f"[CV] Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Expected frame size
        expected_size = (height, width, 3)
        
        # Reset motion gating
        self.motion.reset()
        
        # Storage
        all_events = []
        salient_frames = []
        frame_count = 0
        processed_count = 0
        motion_filtered = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Validate frame
            if frame is None or frame.size == 0:
                log.warning(f"[CV] Skipping invalid frame {frame_count}")
                continue
            
            # Ensure consistent size
            if frame.shape != expected_size:
                log.warning(f"[CV] Frame {frame_count} size mismatch, resizing...")
                frame = cv2.resize(frame, (width, height))
            
            try:
                # Motion gating
                motion_magnitude = self.motion.process(frame)
                
                # Skip low-activity frames
                if motion_magnitude < 2.0:
                    motion_filtered += 1
                    continue
                
                processed_count += 1
                timestamp = frame_count / fps
                
                # Detection
                detections = self.detector.detect(frame)
                
                # Tracking
                tracks = self.tracker.update(detections, frame_count)
                
                # Zone analysis
                zone_data = self.zone_analyzer.analyze(tracks, frame)
                
                # Event generation
                frame_events = self.event_gen.generate_events(tracks, frame_count, timestamp)
                all_events.extend(frame_events)
                
                # Build frame metadata
                persons_data = []
                objects_data = []
                
                for track in tracks:
                    persons_data.append({
                        'track_id': track['track_id'],
                        'bbox': track['bbox'],
                        'confidence': track.get('confidence', 0.0),
                        'zone': track.get('zone', 'unknown'),
                        'motion_state': 'MOVING' if len(track.get('positions', [])) > 5 else 'STATIONARY'
                    })
                
                for det in detections:
                    if det['class'] != 'person':
                        objects_data.append({
                            'class': det['class'],
                            'bbox': det['bbox'],
                            'confidence': det['confidence']
                        })
                
                frame_meta = {
                    'frame_id': frame_count,
                    'timestamp': timestamp,
                    'motion_magnitude': motion_magnitude,
                    'persons': persons_data,
                    'objects': objects_data,
                    'events': frame_events,
                    'zones': zone_data
                }
                
                # Calculate activity score
                activity_score = (
                    len(persons_data) * 10.0 +
                    len(objects_data) * 5.0 +
                    len(frame_events) * 15.0 +
                    motion_magnitude * 2.0
                )
                
                # Store salient frames
                if activity_score > 10.0:
                    salient_frames.append((
                        frame_count,
                        activity_score,
                        frame.copy(),
                        frame_meta
                    ))
                
            except Exception as e:
                log.error(f"[CV] Error processing frame {frame_count}: {e}")
                continue
        
        cap.release()
        
        log.info(f"[CV] Processed {processed_count}/{frame_count} frames ({motion_filtered} filtered)")
        
        # Save all events
        events_path = os.path.join(output_dir, "events.json")
        with open(events_path, "w") as f:
            json.dump(all_events, f, indent=2)
        log.info(f"[CV] Saved {len(all_events)} events to {events_path}")
        
        # Select TOP salient frames by activity
        if not salient_frames:
            log.warning("[CV] No salient frames found!")
            return all_events, []
        
        # Sort by activity score
        salient_frames.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 5 with temporal diversity
        top_salient = []
        seen_ids = set()
        
        for frame_data in salient_frames:
            fid = frame_data[0]
            if fid not in seen_ids:
                # Check temporal diversity (at least 10 frames apart)
                if not top_salient or all(abs(fid - existing[0]) > 10 for existing in top_salient):
                    top_salient.append(frame_data)
                    seen_ids.add(fid)
                    if len(top_salient) >= 5:
                        break
        
        # If not enough diverse frames, just take top 5
        if len(top_salient) < 5:
            for frame_data in salient_frames:
                if frame_data[0] not in seen_ids:
                    top_salient.append(frame_data)
                    seen_ids.add(frame_data[0])
                    if len(top_salient) >= 5:
                        break
        
        log.info(f"[CV] Selected {len(top_salient)} frames with highest activity")
        for idx, (fid, score, _, meta) in enumerate(top_salient, 1):
            log.info(f"  #{idx} Frame {fid}: activity={score:.1f}, {len(meta['persons'])} persons")

        # --- POSE ESTIMATION on salient frames only (no speed penalty on main loop) ---
        enriched_salient = []
        for fid, score, frame_img, meta in top_salient:
            if frame_img is not None:
                for person in meta.get('persons', []):
                    bbox = person.get('bbox', [])
                    if len(bbox) == 4:
                        person['posture'] = self.pose_estimator.estimate(frame_img, bbox)
                    else:
                        person['posture'] = 'unknown'
            enriched_salient.append((fid, score, frame_img, meta))
        top_salient = enriched_salient
        log.info("[CV] ✅ Pose estimation complete on salient frames")
        
        # Save stats
        stats = {
            "video_name": video_name,
            "frames_total": total_frames,
            "frames_processed": processed_count,
            "frames_motion_filtered": motion_filtered,
            "events_detected": len(all_events),
            "salient_frames_selected": len(top_salient)
        }
        
        stats_path = os.path.join(output_dir, "cv_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        
        return all_events, top_salient
    
    def _extract_salient_frames(self, video_path, events, person_tracks, num_frames=4):
        """Extract salient frames - DISABLED, returns empty list."""
        log.info("[CV] _extract_salient_frames called but disabled")
        return []

