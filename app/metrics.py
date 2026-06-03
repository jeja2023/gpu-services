METRICS: dict[str, float] = {
    "requests_total": 0,
    "predict_requests_total": 0,
    "predict_errors_total": 0,
    "model_loads_total": 0,
    "model_load_errors_total": 0,
    "cache_hits_total": 0,
    "cache_misses_total": 0,
    "model_unloads_total": 0,
    "inference_seconds_sum": 0,
    "queue_seconds_sum": 0,
    "model_load_seconds_sum": 0,
    "persons_requests_total": 0,
    "persons_errors_total": 0,
    "persons_detected_total": 0,
    "persons_frames_total": 0,
    "embeddings_requests_total": 0,
    "embeddings_errors_total": 0,
    "tracks_requests_total": 0,
    "tracks_errors_total": 0,
    "vision_requests_total": 0,
    "vision_errors_total": 0,
    "vision_images_total": 0,
    "decode_seconds_sum": 0,
    "preprocess_seconds_sum": 0,
    "postprocess_seconds_sum": 0,
}


def observe(metric: str, value: float = 1) -> None:
    METRICS[metric] = METRICS.get(metric, 0) + value


def metric_label(value: object) -> str:
    return str(value or "").replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def model_labels(model: str, config: dict, extra: dict[str, object] | None = None) -> str:
    rollout = config.get("rollout") if isinstance(config.get("rollout"), dict) else {}
    labels = {
        "model": model,
        "task": config.get("task") or config.get("type") or "",
        "version": config.get("version") or "",
        "status": rollout.get("status") or "",
    }
    if extra:
        labels.update(extra)
    return ",".join(f'{key}="{metric_label(value)}"' for key, value in labels.items())


def prometheus_metrics() -> str:
    from app.model_config import MODEL_CONFIGS
    from app.runtime import MODEL_REGISTRY

    loaded_models = len(MODEL_REGISTRY)
    lines = [
        "# HELP gpu_worker_requests_total Total HTTP requests observed by app middleware.",
        "# TYPE gpu_worker_requests_total counter",
        f"gpu_worker_requests_total {METRICS.get('requests_total', 0)}",
        "# HELP gpu_worker_predict_requests_total Total predict requests.",
        "# TYPE gpu_worker_predict_requests_total counter",
        f"gpu_worker_predict_requests_total {METRICS.get('predict_requests_total', 0)}",
        "# HELP gpu_worker_predict_errors_total Total predict errors.",
        "# TYPE gpu_worker_predict_errors_total counter",
        f"gpu_worker_predict_errors_total {METRICS.get('predict_errors_total', 0)}",
        "# HELP gpu_worker_model_loads_total Total successful model loads.",
        "# TYPE gpu_worker_model_loads_total counter",
        f"gpu_worker_model_loads_total {METRICS.get('model_loads_total', 0)}",
        "# HELP gpu_worker_model_load_errors_total Total failed model loads.",
        "# TYPE gpu_worker_model_load_errors_total counter",
        f"gpu_worker_model_load_errors_total {METRICS.get('model_load_errors_total', 0)}",
        "# HELP gpu_worker_cache_hits_total Total model cache hits.",
        "# TYPE gpu_worker_cache_hits_total counter",
        f"gpu_worker_cache_hits_total {METRICS.get('cache_hits_total', 0)}",
        "# HELP gpu_worker_cache_misses_total Total model cache misses.",
        "# TYPE gpu_worker_cache_misses_total counter",
        f"gpu_worker_cache_misses_total {METRICS.get('cache_misses_total', 0)}",
        "# HELP gpu_worker_model_unloads_total Total model unloads or evictions.",
        "# TYPE gpu_worker_model_unloads_total counter",
        f"gpu_worker_model_unloads_total {METRICS.get('model_unloads_total', 0)}",
        "# HELP gpu_worker_loaded_models Current loaded model count.",
        "# TYPE gpu_worker_loaded_models gauge",
        f"gpu_worker_loaded_models {loaded_models}",
        "# HELP gpu_worker_inference_seconds_sum Sum of inference execution seconds.",
        "# TYPE gpu_worker_inference_seconds_sum counter",
        f"gpu_worker_inference_seconds_sum {METRICS.get('inference_seconds_sum', 0)}",
        "# HELP gpu_worker_queue_seconds_sum Sum of queue wait seconds.",
        "# TYPE gpu_worker_queue_seconds_sum counter",
        f"gpu_worker_queue_seconds_sum {METRICS.get('queue_seconds_sum', 0)}",
        "# HELP gpu_worker_model_load_seconds_sum Sum of model load seconds.",
        "# TYPE gpu_worker_model_load_seconds_sum counter",
        f"gpu_worker_model_load_seconds_sum {METRICS.get('model_load_seconds_sum', 0)}",
        "# HELP gpu_worker_persons_requests_total Total /infer/persons requests.",
        "# TYPE gpu_worker_persons_requests_total counter",
        f"gpu_worker_persons_requests_total {METRICS.get('persons_requests_total', 0)}",
        "# HELP gpu_worker_persons_errors_total Total /infer/persons errors.",
        "# TYPE gpu_worker_persons_errors_total counter",
        f"gpu_worker_persons_errors_total {METRICS.get('persons_errors_total', 0)}",
        "# HELP gpu_worker_persons_detected_total Total detected persons.",
        "# TYPE gpu_worker_persons_detected_total counter",
        f"gpu_worker_persons_detected_total {METRICS.get('persons_detected_total', 0)}",
        "# HELP gpu_worker_persons_frames_total Total frames processed by person detection.",
        "# TYPE gpu_worker_persons_frames_total counter",
        f"gpu_worker_persons_frames_total {METRICS.get('persons_frames_total', 0)}",
        "# HELP gpu_worker_embeddings_requests_total Total /infer/person-embeddings requests.",
        "# TYPE gpu_worker_embeddings_requests_total counter",
        f"gpu_worker_embeddings_requests_total {METRICS.get('embeddings_requests_total', 0)}",
        "# HELP gpu_worker_embeddings_errors_total Total /infer/person-embeddings errors.",
        "# TYPE gpu_worker_embeddings_errors_total counter",
        f"gpu_worker_embeddings_errors_total {METRICS.get('embeddings_errors_total', 0)}",
        "# HELP gpu_worker_tracks_requests_total Total /infer/person-tracks requests.",
        "# TYPE gpu_worker_tracks_requests_total counter",
        f"gpu_worker_tracks_requests_total {METRICS.get('tracks_requests_total', 0)}",
        "# HELP gpu_worker_tracks_errors_total Total /infer/person-tracks errors.",
        "# TYPE gpu_worker_tracks_errors_total counter",
        f"gpu_worker_tracks_errors_total {METRICS.get('tracks_errors_total', 0)}",
        "# HELP gpu_worker_vision_requests_total Total generic /vision inference requests.",
        "# TYPE gpu_worker_vision_requests_total counter",
        f"gpu_worker_vision_requests_total {METRICS.get('vision_requests_total', 0)}",
        "# HELP gpu_worker_vision_errors_total Total generic /vision inference errors.",
        "# TYPE gpu_worker_vision_errors_total counter",
        f"gpu_worker_vision_errors_total {METRICS.get('vision_errors_total', 0)}",
        "# HELP gpu_worker_vision_images_total Total images processed by generic /vision inference.",
        "# TYPE gpu_worker_vision_images_total counter",
        f"gpu_worker_vision_images_total {METRICS.get('vision_images_total', 0)}",
        "# HELP gpu_worker_decode_seconds_sum Sum of image decode seconds.",
        "# TYPE gpu_worker_decode_seconds_sum counter",
        f"gpu_worker_decode_seconds_sum {METRICS.get('decode_seconds_sum', 0)}",
        "# HELP gpu_worker_preprocess_seconds_sum Sum of preprocessing seconds.",
        "# TYPE gpu_worker_preprocess_seconds_sum counter",
        f"gpu_worker_preprocess_seconds_sum {METRICS.get('preprocess_seconds_sum', 0)}",
        "# HELP gpu_worker_postprocess_seconds_sum Sum of postprocessing seconds.",
        "# TYPE gpu_worker_postprocess_seconds_sum counter",
        f"gpu_worker_postprocess_seconds_sum {METRICS.get('postprocess_seconds_sum', 0)}",
        "# HELP gpu_worker_model_config_info Configured model metadata. Value is always 1.",
        "# TYPE gpu_worker_model_config_info gauge",
    ]
    for model, config in sorted(MODEL_CONFIGS.items()):
        lines.append(f"gpu_worker_model_config_info{{{model_labels(model, config)}}} 1")

    lines.extend(
        [
            "# HELP gpu_worker_model_loaded_info Loaded model metadata. Value is always 1.",
            "# TYPE gpu_worker_model_loaded_info gauge",
            "# HELP gpu_worker_model_file_bytes Model artifact file size in bytes.",
            "# TYPE gpu_worker_model_file_bytes gauge",
            "# HELP gpu_worker_model_load_count_total Model load count in this worker.",
            "# TYPE gpu_worker_model_load_count_total counter",
            "# HELP gpu_worker_model_inference_count_total Model inference count in this worker.",
            "# TYPE gpu_worker_model_inference_count_total counter",
            "# HELP gpu_worker_model_last_used_at_seconds Last model use time as Unix timestamp.",
            "# TYPE gpu_worker_model_last_used_at_seconds gauge",
        ]
    )
    for model, bundle in sorted(MODEL_REGISTRY.items()):
        config = MODEL_CONFIGS.get(model, {})
        labels = model_labels(model, config)
        lines.append(f"gpu_worker_model_loaded_info{{{labels}}} 1")
        lines.append(f"gpu_worker_model_file_bytes{{{labels}}} {bundle.get('file_size', 0)}")
        lines.append(f"gpu_worker_model_load_count_total{{{labels}}} {bundle.get('load_count', 0)}")
        lines.append(f"gpu_worker_model_inference_count_total{{{labels}}} {bundle.get('inference_count', 0)}")
        lines.append(f"gpu_worker_model_last_used_at_seconds{{{labels}}} {bundle.get('last_used_at', 0)}")
    return "\n".join(lines) + "\n"
