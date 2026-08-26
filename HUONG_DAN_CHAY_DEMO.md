# Demo kho AGV bằng giọng nói — điền IP rồi chạy

Tài liệu này dành cho người vận hành buổi demo. Chỉ cần **điền một bảng IP ở mục 1**, phần còn
lại chép nguyên lệnh chạy theo đúng máy đang ngồi.

Chi tiết kỹ thuật sâu hơn nằm ở `AIWaiter/HUONG_DAN_DEMO_KHO.md`. File này là bản rút gọn để
chạy được ngay, và đã cập nhật hai điểm mà file kia còn cũ:

- **Toàn bộ đi ZeroTier.** File cũ cho chặng robot đi Netbird — không dùng nữa.
- **Cầu UDP chạy tự động.** File cũ bảo mở terminal thứ hai gõ `python3 -m src.robot_link.bridge`.
  Nay `run_demo.sh` tự chạy cầu đó — chỉ một lệnh, xem mục 5.

---

## 1. BẢNG IP — điền trước khi làm gì khác

Hỏi người dựng mạng ba con số này, hoặc chạy `hostname -I` trên từng máy rồi lấy IP bắt đầu
bằng `172.25.`

| Vai trò | IP ZeroTier | Ghi chú |
|---|---|---|
| **PC server** — LLM + web | `172.25.223.218` | thường không đổi |
| **Máy voice** — mic, loa | `________________` | hôm nay: laptop · mai: Jetson |
| **Máy mô phỏng** — Gazebo | `________________` | máy chạy Docker/Gazebo |

Ba chỗ dùng tới bảng này, tất cả nằm trong **một file `.env` duy nhất trên máy voice**:

```ini
ORCHESTRATOR_URL=http://<IP PC SERVER>:8000
AGENT_URL=http://<IP PC SERVER>:8100
ROBOT_UDP_HOST=<IP MÁY MÔ PHỎNG>
```

**PC server và máy mô phỏng không phải sửa IP gì cả.**

### Nếu voice và mô phỏng là cùng một máy

Đó là trường hợp hôm nay: laptop kiêm cả hai. Khi đó đặt:

```ini
ROBOT_UDP_HOST=127.0.0.1
```

Chạy được vì container Gazebo dùng `--network host`, nên cổng UDP mở thẳng trên máy thật.
Không dùng `--network host` thì phải điền IP ZeroTier của chính laptop thay cho `127.0.0.1`.

### Ngày mai đổi gì

Chỉ **hai dòng** trong `.env`, và `.env` đó chuyển sang nằm trên Jetson:

| Dòng | Hôm nay (laptop kiêm voice) | Ngày mai (Jetson làm voice) |
|---|---|---|
| `ROBOT_UDP_HOST` | `127.0.0.1` | IP ZeroTier của laptop mô phỏng |
| `.env` nằm ở | laptop | Jetson |

`ORCHESTRATOR_URL` và `AGENT_URL` giữ nguyên vì PC server không đổi IP.

---

## 2. Ai làm gì

```
   MÁY VOICE                     PC SERVER                  MÁY MÔ PHỎNG
mic → VAD → Whisper ─ZeroTier─►  agent LLM :8100         Gazebo + Nav2 + V-JEPA
      │                          backend web :8000       cầu UDP nghe :45455
      └──── UDP :45455 qua ZeroTier ───────────────────►  AGV chạy
```

| Máy | Phải cài | Không cần |
|---|---|---|
| PC server | uv, Node 22, Ollama, model 14b | ROS, Gazebo |
| Máy voice | uv, mic + loa, Whisper + Piper | Node, Ollama, ROS |
| Máy mô phỏng | Docker + Gazebo + ROS Jazzy | uv, venv, Node, Ollama |

Máy mô phỏng **không cần venv**: cầu UDP chỉ dùng thư viện chuẩn của Python cộng `rclpy` có sẵn
trong ROS.

---

## 3. PC SERVER

```bash
cd ~/AIWaiter
cp .env.template .env          # không cần sửa dòng nào
ollama serve                   # nếu chưa chạy nền
ollama pull qwen2.5:14b-instruct-q6_K
```

Hai cửa sổ, để chạy suốt buổi:

```bash
make backend      # cửa sổ 1 — port 8000
make agent        # cửa sổ 2 — port 8100
```

Kiểm tra ngay tại chỗ, chưa cần máy khác:

```bash
make checkmap                                        # phải thấy: TẤT CẢ KHỚP
make say TEXT="dẫn tôi đi lấy thùng bia" DRY=1       # in ra lệnh pick_box.sh
```

> Đừng đặt `ROUTER_MODEL` / `WORKER_MODEL` / `RESPONSE_MODEL` / `EMBEDDING_MODEL`. Đó là biến
> của bản nhà hàng cũ, brain kho không đọc tới. Biến sống là `LLM_MODEL`.

---

## 4. MÁY VOICE

```bash
cd ~/AIWaiter
cp .env.template .env
```

Sửa đúng **3 dòng** theo bảng ở mục 1:

```ini
ORCHESTRATOR_URL=http://<IP PC SERVER>:8000
AGENT_URL=http://<IP PC SERVER>:8100
ROBOT_UDP_HOST=<IP MÁY MÔ PHỎNG>
```

Rồi:

```bash
make netcheck     # cả ba dòng phải [ OK ]
make health       # phải thấy: ══ N OK, 0 LỖI ══   (bỏ qua nếu không phải Jetson)
make probe        # nói vào mic, phải in ra text
make voice        # chạy thật
```

Để trống `ROBOT_UDP_HOST` thì máy voice chỉ nghe và trả lời, **không** điều khiển robot — đúng
cho lúc test mic mà chưa mở sa bàn.

---

## 5. MÁY MÔ PHỎNG

Vào trong container Gazebo rồi chạy **một lệnh duy nhất**:

```bash
cd /workshop/warehouse_agv_demo
./run_demo.sh
```

Nó dựng Gazebo + Nav2 + V-JEPA + 5 công nhân, **và tự chạy luôn cầu UDP của AIWaiter**. Không cần
mở terminal thứ hai, không cần biến môi trường nào.

Đợi Gazebo và RViz hiện đủ, rồi kiểm tra:

```bash
./demo_status.sh          # phải thấy: All components ready
```

### Cầu UDP nào đang chạy

Lúc khởi động, dòng đầu của log cầu cho biết nó chọn bản nào:

```bash
head -2 /tmp/warehouse_agv_demo/udp_command_bridge.log
```

Đúng thì thấy:

```
[bridge] AIWaiter: /workshop/AIWaiter/src/robot_link/bridge.py
Nghe lệnh giọng nói trên udp://0.0.0.0:45455
```

Hai repo mỗi bên có một cầu. `run_demo.sh` ưu tiên bản của AIWaiter — đó là bản mà máy voice và
`make say` nhắm tới. Nếu không tìm thấy thư mục AIWaiter, nó lùi về bản đi kèm và in
`[bridge] built-in: ...`. Thấy dòng đó nghĩa là **AIWaiter chưa được clone về đúng chỗ**:

```bash
AIWAITER_DIR=/duong/dan/toi/AIWaiter ./run_demo.sh
```

Muốn ép dùng bản đi kèm: `WAREHOUSE_PREFER_AIWAITER_BRIDGE=false ./run_demo.sh`

### Đừng chạy thêm cầu thứ hai bằng tay

Cả hai cầu đều đặt `SO_REUSEADDR` nên **hai cầu cùng bind được cổng 45455 mà không báo lỗi gì**,
cả hai đều in "listening". Nhưng chỉ **socket bind sau cùng** nhận được gói tin; cầu kia im lặng
nhận 0 gói. Đã đo: gửi 3 gói, cầu bind sau nhận đủ 3, cầu bind trước nhận 0.

Nên đừng chờ thông báo "Address already in use" — không bao giờ có. Kiểm bằng cách đếm:

```bash
pgrep -af "robot_link.bridge|udp_command_bridge"
```

Chỉ được thấy **đúng một dòng**.

### Mở cổng, nếu máy có bật tường lửa

```bash
sudo ufw allow 45455/udp
```

### Lệnh lấy hàng thủ công, để đối chiếu khi cầu UDP có vấn đề

```bash
cd /workshop/warehouse_agv_demo
./pick_box.sh --storage A --color blue --deliver
```

---

## 6. Kiểm tra tăng dần

Mỗi bậc kiểm **một** chặng. Bậc nào hỏng thì sửa xong mới đi tiếp.

| Bậc | Ở đâu | Làm gì | Đúng thì thấy |
|---|---|---|---|
| 1 | PC server | `make checkmap` | `TẤT CẢ KHỚP` |
| 2 | PC server | `make say TEXT="dẫn tôi đi lấy thùng bia" DRY=1` | in ra `pick_box.sh --storage B …` |
| 3 | voice | `make netcheck` | cả ba dòng `[ OK ]` |
| 4 | mô phỏng | `python3 -m src.robot_link.say "dẫn tôi đi lấy thùng bia" --host 127.0.0.1` | xe chạy thật |
| 5 | mô phỏng | `… say "dừng lại" --host 127.0.0.1` rồi `"đi tiếp"` | xe đứng rồi chạy lại |
| 6 | voice | `make probe`, nói vào mic | in ra đúng câu vừa nói |
| 7 | cả ba | bấm **Bắt đầu ra lệnh** trên monitor, nói "kệ nào thiếu đồ" | robot trả lời, **không** chạy đi đâu |

**Bậc 4 và 5 chạy trọn trên máy mô phỏng**, không cần voice lẫn PC server — làm hai bậc này
trước để tách bạch lỗi sa bàn khỏi lỗi mạng.

30 giây trước khi khách vào, trên máy voice:

```bash
make say TEXT="dừng lại"        # phải thấy ✓, máy mô phỏng phải log DỪNG
```

---

## 7. Kịch bản demo

Bấm **Bắt đầu ra lệnh** trên màn monitor trước mỗi câu.

| # | Nói | Robot làm gì |
|---|---|---|
| 1 | "Xin chào" | chào lại — cho thấy nó phân biệt trò chuyện với lệnh |
| 2 | "Kệ nào thiếu đồ?" | đọc tồn kho, **không** chạy đi đâu |
| 3 | "Khu B có gì?" | liệt kê 3 ô kèm màu hộp |
| 4 | "Dẫn tôi đi lấy thùng bia" | **lệnh chính** — chạy tới khu B, gắp hộp xanh, mang về |
| 5 | *đang chạy:* "Dừng lại!" | đứng ngay, ~1,2 giây vì không qua LLM |
| 6 | "Đi tiếp" | chạy tiếp đúng đích cũ, không tính lại đường |

---

## 8. Trục trặc

| Triệu chứng | Nguyên nhân hay gặp |
|---|---|
| Cầu chạy, log "listening", nhưng lệnh không tới | Có **hai** cầu cùng bind 45455. Quên `WAREHOUSE_UDP_COMMAND_BRIDGE=false` ở mục 5 bước 1. Kiểm bằng `pgrep -af "robot_link.bridge\|udp_command_bridge"`, chỉ được một dòng |
| Gửi lệnh không thấy ACK | Sai `ROBOT_UDP_HOST`, hoặc tường lửa chặn 45455/udp |
| Nói được nhưng robot không chạy | `ROBOT_UDP_HOST` để trống trong `.env` máy voice |
| `make netcheck` báo tới được nhưng dịch vụ chưa bật | Chưa chạy `make backend` / `make agent` trên PC server |
| AGV đứng yên sau `run_demo.sh` | Bình thường — `run_demo.sh` chỉ dựng môi trường, phải giao việc riêng |
| Cửa sổ V-JEPA không hiện, lần chạy đầu | Đang tải vài GB thư viện. Xem `./demo_status.sh`, cột `installing deps` |
| Đóng cửa sổ Gazebo là tắt hết | Đúng thiết kế — `gz sim` là tiến trình chính. Đóng RViz thì vô hại |

Kiểm tra cổng 45455 **không dùng** `ss` hay `netstat` — nhiều ảnh Docker không có hai lệnh đó,
chạy sẽ ra kết quả rỗng và tưởng nhầm là cầu đã chết. Dùng:

```bash
cd /workshop/warehouse_agv_demo && ./demo_status.sh
```

---

## 9. Tắt

```bash
# máy mô phỏng: Ctrl+C ở cửa sổ run_demo.sh (tắt luôn mọi thành phần)
# PC server và máy voice:
make kill
```
