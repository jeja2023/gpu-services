from app.metrics import prometheus_metrics


def test_prometheus_metrics_include_model_config_labels() -> None:
    text = prometheus_metrics()

    assert "gpu_worker_model_config_info" in text
    assert 'model="cross_camera_tracking/yolov8n.onnx"' in text
    assert 'task="detection"' in text
    assert 'status="active"' in text
