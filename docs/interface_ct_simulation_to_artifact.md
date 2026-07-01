# 《MedSim_Studio CT Simulation -> CT Artifact 接口规范 v1.2》
文档日期：2026-06-28

## 1. 目的与范围
本文档用于 `MedSim_Studio / ct-simulation` 与 `ct-artifact` 组之间的接口对接，仅描述当前已实现能力。

推荐采用双接口策略：

1. 原始 atlas / phantom CT：`GET /api/v1/simulation/phantom`
2. 参数仿真后的 simulated CT：`POST /api/v1/simulation/ct-params/preview`

## 2. 当前结论

- `/api/v1/simulation/phantom` 支持 `source="atlas"` 与 `source="procedural"`。
- `/api/v1/simulation/ct-params/preview` 支持 `source="atlas"`、`source="procedural"` 与 `source="dicom"`。
- `source="dicom"` 复用 `backend/app/simulation/volume_builder.py` 中的 `build_volume_from_dicom()`，从已上传 DICOM 序列构建 HU volume。
- `ct-params/preview` 当前是 image-domain approximation，不是真实 scanner physics simulation，不用于医学诊断。

## 3. `POST /api/v1/simulation/ct-params/preview`

### 3.1 用途
基于 atlas CT、procedural phantom 或已上传 DICOM CT 生成参数仿真后的 simulated CT volume，用于演示：

- 层厚变化
- 剂量 / mAs 噪声变化
- kVp 对比变化
- pitch 造成的 z 向退化
- FOV 改变
- matrix 分辨率变化
- reconstruction kernel 平滑 / 锐化
- contrast phase 经验增强

### 3.2 请求示例
```json
{
  "source": "dicom",
  "study_id": "<optional-study-id>",
  "series_id": "<series-id>",
  "size": 160,
  "scan_direction": "head_to_feet",
  "params": {
    "slice_thickness_mm": 5.0,
    "dose_level": "low",
    "mAs": 50,
    "kVp": 120,
    "pitch": 1.2,
    "fov_mm": 250,
    "matrix_size": 256,
    "kernel": "bone",
    "contrast_phase": "noncontrast"
  }
}
```

### 3.3 参数表

| 字段 | 类型 | 说明 |
|---|---|---|
| `source` | string | 支持 `atlas / procedural / dicom`；其他值返回 400 |
| `case_id` | string | 仅 `source=atlas` 时使用 |
| `study_id` | string | 仅 `source=dicom` 时使用；提供后会校验 `series_id` 是否属于该 study |
| `series_id` | string | `source=dicom` 时必填；用于定位已上传 DICOM 序列 |
| `size` | int | 当前请求模型限制 `64 <= size <= 192` |
| `scan_direction` | string | `head_to_feet / feet_to_head`；仅 phantom/atlas 方向语义明显 |
| `params.slice_thickness_mm` | number | `0.625 / 1.0 / 2.5 / 5.0 / 10.0` |
| `params.dose_level` | string | `low / standard / high` |
| `params.mAs` | int | `30-300` |
| `params.kVp` | int | `80 / 100 / 120 / 140` |
| `params.pitch` | number | `0.5 / 0.8 / 1.0 / 1.2 / 1.5` |
| `params.fov_mm` | int | `150 / 250 / 350 / 500` |
| `params.matrix_size` | int | `256 / 512 / 1024` |
| `params.kernel` | string | `smooth / soft / standard / lung / bone / sharp` |
| `params.contrast_phase` | string | `noncontrast / arterial / venous / delayed` |

### 3.4 响应要点

- 顶层继续返回 `simulated_volume_base64`、`metadata`、`params_json`、`standardized_case`。
- `standardized_case` 不复制 `simulated_volume_base64`，而是通过 `volume.image_data_field = "simulated_volume_base64"` 指向顶层体数据。
- `source="dicom"` 时，`source_case_id` 通常为 `series_id`。
- `source="dicom"` 时，`origin` / `direction` / `body_part` 优先来自 DICOM metadata；若取不到，会回退到默认值并在 `metadata.notes` 中说明。
- `source="atlas"` 与 `source="procedural"` 仍沿用默认 `origin=[0,0,0]` 与 identity `direction`。

## 4. standardized_case 建议结构

```json
{
  "case_id": "sim_dicom_<series_id>_YYYYMMDD_HHMMSS",
  "source": "dicom",
  "source_case_id": "<series_id>",
  "volume": {
    "encoding": "base64",
    "dtype": "float32",
    "byte_order": "little_endian",
    "axis_order": "zyx",
    "shape": [z, y, x],
    "spacing": [z, y, x],
    "origin": [0.0, 0.0, 0.0],
    "direction": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "hu_range": [min_hu, max_hu],
    "slice_count": z,
    "modality": "CT",
    "body_part": "unknown",
    "image_kind": "simulated_ct",
    "image_data_field": "simulated_volume_base64"
  },
  "simulation": {
    "type": "ct_scan_params",
    "params_json": {},
    "algorithm": "image_domain_approximation",
    "approximation_warning": "This is an educational image-domain approximation, not a scanner-physics simulation."
  },
  "metadata": {}
}
```

## 5. B 组对接方式

### 场景 A：只需要原始 atlas CT
调用：`GET /api/v1/simulation/phantom?source=atlas&case_id=s0001&size=160&scan_direction=head_to_feet`

### 场景 B：需要 atlas / procedural 的参数仿真 CT
调用：`POST /api/v1/simulation/ct-params/preview`

### 场景 C：需要用户上传 DICOM 经过参数仿真后的 CT
调用：`POST /api/v1/simulation/ct-params/preview`

说明：

- `source="dicom"`
- `series_id` 来自 DICOM 上传模块，当前后端实现中必填
- `study_id` 来自 DICOM 上传模块，当前后端实现中可选
- 后端会复用已上传 DICOM 序列，构建 HU volume 后再进入 CT 参数仿真

## 6. 当前未实现

- NRRD / NIfTI / DICOM 文件导出
- MinIO / 数据库登记新流程
- B 组 artifact pipeline 返回展示
- simulated volume 3D rendering
- 更真实的物理仿真
- 完整异步 job / volume / labels / series 资源化接口

## 7. 错误格式与注意事项

- 业务错误通常为 `{"detail": "..."}`。
- `source` 若不是 `atlas`、`procedural` 或 `dicom`，返回 400。
- `source="dicom"` 且缺少 `series_id`，返回 400。
- 找不到 study / series 或实例列表为空时，通常返回 404。
- B 组解码体数据时，必须按 `(z, y, x)` 和 little-endian `float32` 解释 `simulated_volume_base64`。
