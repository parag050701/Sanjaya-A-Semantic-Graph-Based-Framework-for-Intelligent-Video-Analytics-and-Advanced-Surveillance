from flask import Flask, request, jsonify, send_from_directory, render_template, url_for, Response
from flask_cors import CORS
from flask_socketio import SocketIO
import os
import logging
import cv2
import json
import csv
import io
import shutil
import sys
import threading
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from cv_pipeline.pipeline import CVPipeline
from modules.vlm_analyzer import analyze_salient_frame
from rag.json_rag import JsonRAG
from rag.graph_rag import GraphRAG
from modules.neo4j_kg import export_surveillance_graph, push_vlm_kg_to_neo4j, push_vlm_analysis_summary
from modules.neo4j_manager import ensure_neo4j

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger("__main__")

# ─── Flask + SocketIO ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'sanjaya-secret'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ─── Directories ──────────────────────────────────────────────────────────────
UPLOAD_DIR = "uploads"
FRAMES_DIR = "static/frames"
JSON_DIR   = "json_outputs"
for d in [UPLOAD_DIR, FRAMES_DIR, JSON_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Session history (in-memory + persisted to sessions.json) ─────────────────
SESSIONS_FILE = os.path.join(JSON_DIR, "sessions.json")

def _load_sessions():
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []

def _save_sessions(sessions):
    with open(SESSIONS_FILE, 'w') as f:
        json.dump(sessions, f, indent=2)

# ─── Component init ───────────────────────────────────────────────────────────
log.info("[INIT] Initializing pipeline components...")

class Neo4jKG:
    def __init__(self):
        self.uri  = "bolt://localhost:7687"
        self.auth = ("neo4j", "neo4j123")
        ensure_neo4j()

    def push_cv_events(self, events, video_id="current_video"):
        events_path = os.path.join(JSON_DIR, "cv_events_temp.json")
        with open(events_path, 'w') as f:
            json.dump(events, f)
        return export_surveillance_graph(self.uri, self.auth, events_path, video_id)

    def push_vlm_kg(self, frame_id, kg_data, video_id="current_video"):
        return push_vlm_kg_to_neo4j(self.uri, self.auth, kg_data, video_id, frame_id)

cv_pipeline      = CVPipeline()
rag_engine       = JsonRAG(json_dirs=[JSON_DIR])
graph_rag_engine = GraphRAG()
neo4j_kg         = Neo4jKG()
log.info("[INIT] ✅ All components initialized")

# ─── Progress emit helper ─────────────────────────────────────────────────────
def emit_progress(stage: str, pct: int, msg: str):
    socketio.emit('progress', {'stage': stage, 'pct': pct, 'msg': msg})
    log.info(f"[PROGRESS] {pct}% — {stage}: {msg}")

# ─── YOLO annotation helper ───────────────────────────────────────────────────
def draw_yolo_annotations(frame, persons, objects_list):
    annotated = frame.copy()
    for p in persons:
        bbox = p.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            track_id = p.get('track_id', '?')
            speed    = p.get('speed_px_s', 0)
            posture  = p.get('posture', 'standing')
            motion   = p.get('motion_state', 'unknown')
            zone     = p.get('zone', '?')
            label    = f"P{track_id} Z{zone} {speed:.0f}px/s {motion} [{posture}]"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, y1 - h - 4), (x1 + w, y1), (0, 200, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    for o in objects_list:
        bbox = o.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 100, 0), 2)
            obj_class = o.get('class', 'object')
            conf      = o.get('confidence', 0)
            zone      = o.get('zone', '?')
            label     = f"{obj_class} Z{zone} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, y1 - h - 4), (x1 + w, y1), (255, 100, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return annotated

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/test_events")
def test_events():
    return render_template("test_events.html")

@app.route("/api/videos")
def list_videos():
    try:
        videos = [f for f in os.listdir(UPLOAD_DIR)
                  if f.endswith(('.mp4', '.avi', '.mov'))]
        return jsonify({"videos": videos})
    except Exception:
        return jsonify({"videos": []})

@app.route("/api/sessions")
def list_sessions():
    return jsonify({"sessions": _load_sessions()})

# ─── Main pipeline ────────────────────────────────────────────────────────────
@app.route("/pipeline/upload", methods=["POST"])
def pipeline_upload():
    try:
        # 1. SAVE VIDEO
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        video_id   = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        video_path = os.path.join(UPLOAD_DIR, file.filename)
        file.save(video_path)
        emit_progress("upload", 5, f"Saved {file.filename}")

        # 2. CLEAR FRAME / JSON OUTPUTS (keep sessions.json and previous graph)
        for f in os.listdir(JSON_DIR):
            if f.endswith('.json') and f != 'sessions.json':
                os.remove(os.path.join(JSON_DIR, f))
        for f in os.listdir(FRAMES_DIR):
            if f.endswith('.jpg'):
                os.remove(os.path.join(FRAMES_DIR, f))
        emit_progress("cleanup", 8, "Cleared previous outputs")

        # 3. CV PIPELINE
        emit_progress("cv_pipeline", 10, "Starting motion gating + detection + tracking…")
        cv_events, salient_frames_data = cv_pipeline.process_video(
            video_path=video_path,
            output_dir=JSON_DIR
        )
        emit_progress("cv_pipeline", 45, f"{len(cv_events)} events · {len(salient_frames_data)} salient frames")

        # FALLBACK: middle frame
        if len(salient_frames_data) == 0:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps          = cap.get(cv2.CAP_PROP_FPS) or 30
            mid          = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, mf = cap.read()
            cap.release()
            if ret and mf is not None:
                salient_frames_data = [(mid, 0.5, mf, {
                    'frame_id': mid, 'timestamp': mid / fps,
                    'saliency': 0.5, 'persons': [], 'objects': [],
                    'all_objects': [], 'zones': {}
                })]

        # 4. ANNOTATE TOP-3 FRAMES
        salient_frames = []
        for idx, (frame_id, saliency_score, frame_img, cv_metadata) in enumerate(salient_frames_data[:3]):
            fname      = f"salient_{idx}_frame{frame_id}.jpg"
            fpath      = os.path.join(FRAMES_DIR, fname)
            persons    = cv_metadata.get('persons', [])
            objects_list = cv_metadata.get('objects', [])
            if frame_img is not None:
                annotated = draw_yolo_annotations(frame_img, persons, objects_list)
                cv2.imwrite(fpath, annotated)
            cv_metadata["image_path"] = os.path.abspath(fpath)
            cv_metadata["image_url"]  = url_for('static', filename=f'frames/{fname}')
            salient_frames.append(cv_metadata)
        emit_progress("frames", 50, f"Annotated {len(salient_frames)} frames (with pose labels)")

        # 5. VLM ANALYSIS
        vlm_results = []
        for idx, frame_meta in enumerate(salient_frames[:3]):
            frame_id  = frame_meta['frame_id']
            frame_path = frame_meta['image_path']
            timestamp  = frame_meta.get('timestamp', 0)
            emit_progress("vlm", 52 + idx * 10, f"VLM analysing frame {idx+1}/3…")
            try:
                cv_detections = {
                    "persons": frame_meta.get("persons", []),
                    "objects": frame_meta.get("objects", [])
                }
                vlm_result = analyze_salient_frame(
                    image_path=frame_path,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    cv_detections=cv_detections,
                    ollama_url="http://localhost:11434"
                )
                vlm_results.append(vlm_result)

                with open(os.path.join(JSON_DIR, f"frame_{frame_id}_vlm.json"), 'w') as f:
                    json.dump(vlm_result, f, indent=2)
                with open(os.path.join(JSON_DIR, f"frame_{frame_id}_cv.json"), 'w') as f:
                    json.dump({
                        "frame_id": frame_id, "timestamp": timestamp,
                        "image_path": frame_path,
                        "persons": frame_meta.get("persons", []),
                        "objects": frame_meta.get("objects", []),
                        "zones":   frame_meta.get("zones", {}),
                        "motion_magnitude": frame_meta.get("motion_magnitude", 0)
                    }, f, indent=2)
            except Exception as e:
                log.error(f"[VLM] Failed frame {frame_id}: {e}")
        emit_progress("vlm", 82, f"VLM complete — {len(vlm_results)}/3 frames analysed")

        # 6. NEO4J — PERSISTENT (video_id keeps sessions separate in graph)
        emit_progress("neo4j", 84, "Pushing to knowledge graph…")
        if cv_events:
            try:
                neo4j_kg.push_cv_events(cv_events, video_id)
            except Exception as e:
                log.error(f"[Neo4j] CV push failed: {e}")
        for vlm_result in vlm_results:
            try:
                fid = vlm_result.get('frame_id', 'unknown')
                if 'knowledge_graph' in vlm_result and vlm_result['knowledge_graph']:
                    neo4j_kg.push_vlm_kg(fid, vlm_result['knowledge_graph'], video_id)
                push_vlm_analysis_summary(neo4j_kg.uri, neo4j_kg.auth,
                                          vlm_result, video_id, fid)
            except Exception as e:
                log.error(f"[Neo4j] VLM push failed: {e}")
        emit_progress("neo4j", 88, "Knowledge graph updated")

        # 7. RAG INDEX
        emit_progress("rag", 90, "Building RAG index…")
        rag_engine.build_index()
        emit_progress("rag", 94, f"RAG indexed {len(rag_engine.documents)} documents")

        # 8. AGGREGATE INSIGHTS
        all_risks, all_anomalies, all_objects = [], [], set()
        for vlm in vlm_results:
            all_risks.extend(vlm.get("risks", []))
            all_anomalies.extend(vlm.get("anomalies", []))
        for frame in salient_frames[:3]:
            for obj in frame.get("objects", []):
                if obj.get("class"):
                    all_objects.add(obj["class"])

        risk_levels = [r.get("severity", "low") for r in all_risks]
        overall_risk = ("high"   if risk_levels.count("high") > 0 else
                        "medium" if risk_levels.count("medium") > 0 else "low")

        # collect posture summary
        posture_counts = {}
        for frame in salient_frames[:3]:
            for p in frame.get("persons", []):
                pos = p.get("posture", "unknown")
                posture_counts[pos] = posture_counts.get(pos, 0) + 1

        # 9. SAVE SESSION METADATA
        sessions = _load_sessions()
        session_entry = {
            "video_id":    video_id,
            "filename":    file.filename,
            "timestamp":   datetime.now().isoformat(),
            "event_count": len(cv_events),
            "risk_level":  overall_risk,
            "frame_count": len(salient_frames),
            "posture_summary": posture_counts,
            "detected_objects": list(all_objects),
        }
        sessions.insert(0, session_entry)
        sessions = sessions[:20]   # keep last 20
        _save_sessions(sessions)

        emit_progress("done", 100, f"Analysis complete — {len(cv_events)} events · risk: {overall_risk}")

        return jsonify({
            "status":        "success",
            "video_id":      video_id,
            "salient_frames": salient_frames,
            "vlm_results":   vlm_results,
            "cv_events":     len(cv_events),
            "insights": {
                "risks":            all_risks,
                "anomalies":        all_anomalies,
                "overall_risk":     overall_risk,
                "detected_objects": list(all_objects),
                "posture_summary":  posture_counts,
            },
        })

    except Exception as e:
        log.error(f"[PIPELINE] Error: {e}", exc_info=True)
        socketio.emit('progress', {'stage': 'error', 'pct': 0, 'msg': str(e)})
        return jsonify({"error": str(e)}), 500


# ─── Export endpoint ──────────────────────────────────────────────────────────
@app.route("/api/export/report")
def export_report():
    """Download incident report as JSON or CSV."""
    fmt = request.args.get("format", "json").lower()

    # Gather data
    events = []
    events_path = os.path.join(JSON_DIR, "events.json")
    if os.path.exists(events_path):
        with open(events_path) as f:
            events = json.load(f)

    vlm_data = []
    cv_data  = []
    for fname in os.listdir(JSON_DIR):
        if fname.endswith("_vlm.json"):
            with open(os.path.join(JSON_DIR, fname)) as f:
                vlm_data.append(json.load(f))
        elif fname.endswith("_cv.json"):
            with open(os.path.join(JSON_DIR, fname)) as f:
                cv_data.append(json.load(f))

    stats = {}
    stats_path = os.path.join(JSON_DIR, "cv_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)

    report = {
        "generated_at":   datetime.now().isoformat(),
        "platform":       "Sanjaya Video Intelligence Platform",
        "cv_stats":       stats,
        "total_events":   len(events),
        "events":         events,
        "vlm_analyses":   vlm_data,
        "cv_frame_data":  cv_data,
    }

    if fmt == "csv":
        si = io.StringIO()
        writer = csv.DictWriter(si, fieldnames=[
            "type", "track_id", "frame_id", "timestamp",
            "motion_state", "speed_px_s", "zone", "priority"
        ])
        writer.writeheader()
        for ev in events:
            writer.writerow({k: ev.get(k, "") for k in writer.fieldnames})
        output = si.getvalue()
        return Response(
            output,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=sanjaya_report.csv"}
        )
    else:
        return Response(
            json.dumps(report, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": "attachment; filename=sanjaya_report.json"}
        )


# ─── RAG endpoints ────────────────────────────────────────────────────────────
@app.route("/rag/search")
def rag_search():
    q = request.args.get("q", "").strip()
    k = int(request.args.get("k", 5))
    if not q:
        return jsonify({"error": "empty query"}), 400
    try:
        result = rag_engine.ask(q, k=k)
        return jsonify({
            "answer":     result["answer"],
            "evidence":   result["evidence"],
            "confidence": result["confidence"],
            "sources":    result["sources"],
            "type":       "json_rag"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/rag/graph")
def graph_rag_search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "empty query"}), 400
    try:
        result = graph_rag_engine.ask(q)
        return jsonify({
            "answer":          result.get("answer"),
            "chain_of_thought": result.get("chain_of_thought", []),
            "evidence":        result.get("evidence", []),
            "confidence":      result.get("confidence", 0.0),
            "reasoning_path":  result.get("reasoning_path", ""),
            "graph_facts":     result.get("graph_facts", []),
            "type":            "graph_rag"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/rag/hybrid")
def hybrid_rag_search():
    q = request.args.get("q", "").strip()
    k = int(request.args.get("k", 3))
    if not q:
        return jsonify({"error": "empty query"}), 400
    try:
        graph_result = graph_rag_engine.ask(q)
        json_result  = rag_engine.ask(q, k=k)
        combined = (f"**Graph Reasoning:** {graph_result.get('answer')}\n\n"
                    f"**JSON Evidence:** {json_result['answer']}")
        json_conf_str = json_result["confidence"].rstrip('%')
        json_conf = float(json_conf_str) / 100 if json_conf_str else 0.0
        return jsonify({
            "answer": combined.strip(),
            "graph_reasoning": {
                "chain_of_thought": graph_result.get("chain_of_thought", []),
                "reasoning_path":   graph_result.get("reasoning_path", "")
            },
            "json_evidence": {
                "evidence": json_result["evidence"],
                "sources":  json_result["sources"]
            },
            "confidence": (graph_result.get("confidence", 0) + json_conf) / 2,
            "type": "hybrid_rag"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Static / utility ─────────────────────────────────────────────────────────
@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)

@app.route("/json_outputs/<path:filename>")
def serve_json_outputs(filename):
    return send_from_directory("json_outputs", filename)

@app.route("/api/telegram/status")
def telegram_status():
    return jsonify({"active": False, "message": "Telegram disabled"})

@app.route("/rag/status")
def rag_status():
    status = {
        "index_built":    rag_engine.index is not None,
        "document_count": len(rag_engine.documents),
        "json_dirs":      rag_engine.json_dirs,
        "json_files":     []
    }
    for json_dir in rag_engine.json_dirs:
        if os.path.exists(json_dir):
            status["json_files"].extend(
                [f for f in os.listdir(json_dir) if f.endswith('.json')]
            )
    return jsonify(status)

@app.route("/rag/rebuild", methods=["POST"])
def rebuild_rag():
    try:
        rag_engine.build_index()
        return jsonify({
            "status":         "success",
            "document_count": len(rag_engine.documents),
            "index_ready":    rag_engine.index is not None
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ─── SocketIO events ──────────────────────────────────────────────────────────
@socketio.on('connect')
def handle_connect():
    log.info("[SocketIO] Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    log.info("[SocketIO] Client disconnected")


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
