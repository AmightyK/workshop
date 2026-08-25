# Hướng dẫn chạy AGV và định vị V-JEPA

Tài liệu này mô tả đúng hai lượt chạy:

1. **Mapping:** Nav2 lái AGV qua toàn bộ waypoint, camera ghi video và pose
   truth của Gazebo. V-JEPA trích latent và lưu thành visual map.
2. **Query:** Nav2 lái lại cùng tuyến đường. V-JEPA chỉ dùng camera để dự đoán
   pose; pose world của Gazebo chỉ dùng sau đó để chấm điểm.

Không có GPS/Gazebo pose nào được đưa vào node V-JEPA khi suy luận.

## 0. Chạy demo pick box hằng ngày

Không cần chạy lại mapping để pick box. Mở hai terminal:

```bash
# Terminal 1
cd /home/amightyk05/workshop/warehouse_agv_demo
./run_demo.sh

# Terminal 2
cd /home/amightyk05/workshop/warehouse_agv_demo
./pick_box.sh --area A --color blue
```

`run_demo.sh` tự mở Gazebo, bridge, người đi bộ, V-JEPA temporal và hai cửa sổ
so sánh. `pick_box.sh` tự mở Nav2 nếu chưa chạy, đi theo tuyến vào tủ rồi mới
pick màu đã chọn. Có thể dùng mọi tổ hợp `A/B/C` và `red/blue/green`:

```bash
./pick_box.sh --area B --color red --deliver
./pick_box.sh --area C --color green --route-only
./pick_box.sh --area A --color red --dry-run
```

Log của bridge/V-JEPA/dashboard nằm trong `/tmp/warehouse_agv_demo`.

### Demo giống video VL-JEPA với một hoặc hai tuyến cố định

Sau khi `run_demo.sh` đã mở, terminal thứ hai chọn tuyến:

```bash
./run_vljepa_showcase.sh --route short  # dock -> tủ A
./run_vljepa_showcase.sh --route long   # dock -> tủ C, qua hai worker
./run_vljepa_showcase.sh --route both   # chạy lần lượt cả hai
```

Cửa sổ thứ nhất hiển thị camera 16:9 bên trái và latent PCA bên phải. Đám mây
xám là latent map đã lưu, chấm đỏ là embedding camera hiện tại, chấm xanh là
latent được temporal tracker chấp nhận. Tám câu hỏi soạn sẵn tự đổi mỗi 3.5
giây; sửa nội dung/thứ tự tại `configs/warehouse_live_questions.yaml`.

Mỗi latent dùng rolling window gồm 4 ảnh mới nhất ở 4 FPS trong 1 giây. Trong
khi GPU inference, callback vẫn cập nhật ảnh và bỏ frame cũ nên lần kế tiếp
không xử lý backlog. Pose được đóng dấu ở tâm clip, vì vậy độ trễ RAW trung thực
là 0.5 giây cộng thời gian inference.
Dashboard mặc định không subscribe `/odom`: vị trí vàng là camera-only. Chỉ
bật `--odom-projection` khi cần thêm pose dự phóng vào JSON telemetry; pose đó
không thay thế `V-JEPA RAW` trên bản đồ.

Câu trả lời là template chuẩn bị sẵn được điền bằng trạng thái V-JEPA, A*,
LiDAR và evaluator Gazebo. Đây là cách trình bày theo video VL-JEPA; checkpoint
V-JEPA 2 hiện tại không được giả là một language decoder end-to-end.

## Temporal prior hoạt động như thế nào

Mỗi kết quả mới phải hợp lý so với pose **V-JEPA trước đó** về khoảng cách,
góc quay và thứ tự latent đã ghi. Nó không lấy odometry hay GPS Gazebo làm
prior. Feature tracking và essential geometry trên hai ảnh liên tiếp chỉ báo
khi camera thực sự đang tiến; quay tại chỗ để né người không được phép kéo pose
sang một kệ giống hình ở xa.

- `TRACKING`: nhận một latent gần, hợp lệ và làm mượt pose.
- `HOLDING`: ảnh đang mơ hồ; giữ pose trước thay vì nhảy lung tung.
- `RELOCALIZED`: chỉ nhận lại vị trí sau nhiều frame nhất quán và vẫn giới hạn
  khoảng nhảy.
- `FORWARD` / `TURN-STILL`: tín hiệu chuyển động được suy ra hoàn toàn từ ảnh.

Các ngưỡng nằm trong `configs/warehouse_experiment.yaml` ở mục
`temporal_tracking`.

## 1. Chạy đầy đủ và lưu latent

Từ thư mục dự án:

```bash
cd /home/amightyk05/workshop/vjepa_visual_localization
./run_autonomous_experiment.sh --launch-stack --headless --overwrite
```

Lệnh trên tự chạy Gazebo, bridge, Nav2 và bốn pha mapping, tạo latent, query,
đánh giá. Các worker được đưa ra ngoài khi mapping và được bật lại khi query.

Latent được lưu bền vững tại:

```text
outputs/autonomous_map_dense/global_embeddings.npy  # [751, 1024], float16
outputs/autonomous_map_dense/poses.npy              # truth gắn với từng latent map
outputs/autonomous_map_dense/timestamps.npy
outputs/autonomous_map_dense/ids.npy
outputs/autonomous_map_dense/metadata.json          # config + thông tin camera
```

Không xóa thư mục `outputs/autonomous_map_dense` nếu muốn dùng lại vào hôm sau.

## 2. Dùng lại latent, không chạy mapping lần nữa

```bash
./run_autonomous_experiment.sh \
  --launch-stack --headless --overwrite --reuse-latents
```

`--reuse-latents` là tên dễ nhớ của `--skip-mapping`. Chế độ này chỉ chạy
query và evaluation bằng visual map đã lưu. Phải tạo lại map nếu đổi góc camera,
độ phân giải, checkpoint V-JEPA hoặc bố cục kho.

## 3. AGV phải di chuyển khi so sánh truth

Tuyến kín mặc định bao phủ đúng đường pick A/B/C:

```text
charging_dock -> east_cross_aisle -> central_south
              -> A staging/front -> B staging/front -> C staging/front
              -> central_return -> south_center -> charging_dock
```

`evaluation.min_translation_m: 0.10` khiến cả đánh giá live và offline bỏ các
clip mà AGV dịch chuyển dưới 10 cm. Vì vậy các mẫu lúc chờ Nav2, dừng ở waypoint
hoặc đứng tại dock không được tính vào metric. Log sẽ báo:

```text
[LOCALIZE_SKIP] AGV đang đứng yên; không đưa mẫu này vào so sánh V-JEPA/Gazebo
[LOCALIZE_RESUME] AGV đã di chuyển; tiếp tục so sánh V-JEPA với truth Gazebo
```

Mỗi dòng so sánh chuyển động chứa pose truth, pose V-JEPA, độ dịch chuyển thật
trong cửa sổ thời gian và sai số tương đối:

```text
[LOCALIZE] khu kệ T | Gazebo=(2.89,-13.62) V-JEPA=(3.02,-13.80)
           lệch tương đối=(+0.13,-0.18) m, lỗi=0.22 m
```

## 4. Camera và tỉ lệ ảnh

Camera trên AGV có pitch bằng `0 rad`, tức trục quang học song song mặt đất.

Camera Gazebo phát ảnh raw `640x360`, tỉ lệ `16:9`; ảnh raw **không vuông**.
Processor của `facebook/vjepa2-vitl-fpc64-256` resize theo cạnh ngắn rồi
center-crop thành `256x256`, tỉ lệ `1:1`. Cách này không kéo méo ảnh, nhưng cắt
một phần hai cạnh trái/phải.

Kiểm tra lại video bất kỳ bằng `uv`:

```bash
uv run python scripts/inspect_camera_pipeline.py \
  --run data/autonomous_query_dense --expected-aspect 16:9
```

Kết quả được lưu vào `data/autonomous_query_dense/camera_pipeline.json`. Metadata raw
từ ROS nằm tại `camera_metadata.json` trong cùng thư mục.

## 5. Đọc kết quả

```text
outputs/autonomous_experiment_dense/query_comparison.csv
outputs/autonomous_experiment_dense/query_events.jsonl
outputs/autonomous_experiment_dense/query_summary.json
outputs/autonomous_experiment_dense/offline_predictions.jsonl
outputs/autonomous_experiment_dense/offline_metrics.json
outputs/autonomous_experiment_dense/experiment_summary.json
```

- `query_comparison.csv`: chỉ các phép so sánh khi AGV đang dịch chuyển.
- `offline_predictions.jsonl`: top-k latent, truth, dự đoán và sai số từng clip.
- `offline_metrics.json`: mean/median/P95 và Recall theo khoảng cách.
- `query_events.jsonl`: khu vực hiện tại, có/không có vật cản và đang né gì.

Ví dụ sự kiện tránh vật cản:

```text
[STATUS] khu đóng gói | vật cản=có: thùng tĩnh 4 (1.82 m)
[AVOID] đang rẽ phải để né thùng tĩnh 4 (LiDAR 1.82 m)
[AVOID] đang rẽ trái để né người đi bộ 1 (LiDAR 1.35 m)
```

LiDAR quyết định có vật cản hay không. Tên model Gazebo chỉ được evaluator dùng
để chú thích “thùng tĩnh” hoặc “người đi bộ”; V-JEPA không nhận thông tin này.

## 6. Mở hai cửa sổ so sánh trực tiếp

Chạy query có giao diện và dùng lại latent đã lưu:

```bash
./run_autonomous_experiment.sh \
  --launch-stack --overwrite --reuse-latents
```

Pha query tự mở hai cửa sổ:

1. `VL-JEPA Warehouse Streaming QA`: chỉ giữ header query/model, camera 16:9 và
   latent PCA đỏ/xanh như hình tham chiếu thứ hai; hai card telemetry phía dưới
   đã được ẩn. Focus cửa sổ rồi giữ `W/S` để tiến-lùi và giữ thêm `A/D` cùng lúc
   để quẹo; chỉ giữ `A/D` thì quay tại chỗ. Thả phím hoặc nhấn `Space` để dừng,
   `Esc` để đóng.
2. `Warehouse Map - Truth GPS & Planning`: bản đồ LiDAR chỉ với vệt/mũi tên
   Truth GPS và đường planning hiện hành. V-JEPA không được vẽ trong map 2D;
   phần V-JEPA vẫn nằm ở cửa sổ streaming/latent.

Terminal mission vẫn dùng card màu, progress bar waypoint và các dòng
`OBSTACLE`, `AVOID`, `CLEAR` thay cho JSON thô.

Nếu localizer và Gazebo đã chạy sẵn, chỉ mở dashboard bằng:

```bash
./run_localization_dashboard.sh
```

Dashboard là node đánh giá tách biệt. Node V-JEPA chỉ subscribe topic ảnh DDS
`/vjepa/camera/image_raw`; truth Gazebo không được đưa vào encoder hoặc bộ truy
hồi latent. Khi chạy V-JEPA trên Orin qua dây LAN trực tiếp, làm theo
[`ORIN_DDS_LAN.md`](ORIN_DDS_LAN.md).

## 7. Chạy test

```bash
uv run pytest -q
uv lock --check
```

Nếu Nav2 tạm abort vì TF chậm hơn clock Gazebo một nhịp, patrol tự chờ hai giây
và thử lại waypoint. Không cần điều khiển AGV bằng tay.
