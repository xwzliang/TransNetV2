import os
import io
import tempfile
import numpy as np
import gc
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional
from PIL import Image

from transnetv2_pytorch_runner import TransNetV2
from pydantic import BaseModel
from logging_setup import get_logger
import threading

logger = get_logger()
# ----- FastAPI server -----

app = FastAPI(
    title="TransNetV2 Inference API",
    debug=True,
)
_model: Optional[TransNetV2] = None


@app.post("/self-shutdown")
async def self_shutdown():
    def exit_later():
        # short delay to ensure response is sent
        import time

        time.sleep(0.1)
        os._exit(0)  # immediate hard exit

    threading.Thread(target=exit_later, daemon=True).start()
    return {"status": "shutting down"}


@app.post("/load_model")
def load_model():
    global _model
    _model = TransNetV2()


@app.post("/unload_model")
def unload_model():
    global _model
    if _model is None:
        return JSONResponse({"detail": "Model already unloaded."})
    _model = None
    gc.collect()
    logger.info("Model unloaded from memory")
    return JSONResponse({"detail": "Model unloaded successfully."})


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/infer")
async def predict(
    file: UploadFile = File(..., description="Video file to analyze"),
    visualize: bool = Form(False, description="Return visualization image if true"),
    threshold: float = Form(0.5, description="Scene detection threshold"),
):
    if _model is None:
        load_model()

    # save upload to temp file
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename)[1]
    ) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        video, single_pred, all_pred = _model.predict_video(tmp_path)
        scenes = _model.predictions_to_scenes(single_pred, threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if visualize:
        img = _model.visualize_predictions(video, (single_pred, all_pred))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    return JSONResponse(
        {
            "single_frame_predictions": single_pred.tolist(),
            "all_frame_predictions": all_pred.tolist(),
            "scenes": scenes.tolist(),
        }
    )


class PredictPathRequest(BaseModel):
    video_path: str
    visualize: Optional[bool] = False
    threshold: Optional[float] = 0.5


@app.post("/infer_path")
def predict_path(req: PredictPathRequest):
    logger.info(f"Get request: {req}")
    if _model is None:
        load_model()

    if not os.path.isfile(req.video_path):
        raise HTTPException(status_code=400, detail=f"File not found: {req.video_path}")

    try:
        video, single_pred, all_pred = _model.predict_video(req.video_path)
        scenes = _model.predictions_to_scenes(single_pred, req.threshold)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if req.visualize:
        img = _model.visualize_predictions(video, (single_pred, all_pred))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    return JSONResponse(
        {
            "single_frame_predictions": single_pred.tolist(),
            "all_frame_predictions": all_pred.tolist(),
            "scenes": scenes.tolist(),
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True
    )
