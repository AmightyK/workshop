# Kết nối AIWaiter ↔ AGV bằng UDP

Tài liệu này là hợp đồng kết nối giữa máy AIWaiter (máy nhận giọng nói) và
`warehouse_agv_demo` (máy chạy Gazebo/Nav2). AIWaiter chỉ gửi JSON qua UDP;
không cài ROS và không truy cập DDS của AGV.

## 1. Phía AGV khởi động trước

Trên máy chạy AGV:

```bash
cd /home/amightyk05/workshop/warehouse_agv_demo
./run_demo.sh
```

`run_demo.sh` tự khởi động cầu UDP và lắng nghe trên:

```text
udp://0.0.0.0:45455
```

Phải thấy dòng tương tự:

```text
AIWaiter UDP       : listening on 0.0.0.0:45455
```

Nếu máy AGV có firewall:

```bash
sudo ufw allow 45455/udp
```

Lấy IP mà máy AIWaiter có thể đi tới:

```bash
hostname -I
```

Nếu kết nối qua NetBird/ZeroTier, dùng IP của đúng mạng đó, không dùng
`127.0.0.1` (trừ khi hai chương trình chạy cùng một máy).

## 2. Phía AIWaiter cấu hình địa chỉ đích

Trong file `.env` của AIWaiter, đặt:

```ini
ROBOT_UDP_HOST=<IP_MAY_AGV>
ROBOT_UDP_PORT=45455
ROBOT_LINK_SEND_ACTION=1
```

Ví dụ:

```ini
ROBOT_UDP_HOST=100.66.149.248
ROBOT_UDP_PORT=45455
```

`ROBOT_UDP_HOST` là bắt buộc để bật gửi lệnh. Để trống biến này thì AIWaiter
vẫn trả lời bằng giọng nói nhưng không gửi lệnh xuống AGV.

AIWaiter đã có sẵn `src/robot_link/protocol.py` và
`src/robot_link/sender.py`; nên dùng `CommandSender`/`build_sender()` thay vì
tự tạo một giao thức khác.

## 3. Lệnh di chuyển tới kệ A, lấy hộp xanh và mang về

Lệnh chuẩn mà AIWaiter gửi là một datagram JSON v1:

```json
{
  "v": 1,
  "kind": "navigate",
  "action": {
    "type": "navigate",
    "position": {
      "token": "A",
      "section": "A",
      "slot": "A01",
      "color": "blue"
    },
    "task": "fetch"
  },
  "sentence": "tới kệ A lấy hộp xanh rồi mang về",
  "reply": "Đang tới kệ A lấy hộp xanh và mang về.",
  "source": "agent",
  "robot_id": "robo-1",
  "session": "ca8f31d2",
  "seq": 1,
  "ts": 1730000000.0
}
```

Trong đó:

- `position.token` phải là `A`. AGV hiện chỉ thực hiện Storage A.
- `position.color` nhận `blue`, `red` hoặc `green`; nếu bỏ trống thì mặc định
  là `blue` ở cầu AGV.
- `task: "fetch"` nghĩa là đi tới kệ, gắp hộp, rồi mang về trạm đóng gói.
- `task: "fetch_hold"` nghĩa là gắp xong giữ hộp trên khay, không chạy chặng
  mang về.
- `session` là mã một phiên gửi; `seq` tăng dần trong phiên. Không được dùng
  lại cùng cặp `(session, seq)` cho hai lệnh khác nhau.

Cách gọi bằng sender có sẵn:

```python
from src.robot_link.sender import build_sender

sender = build_sender(robot_id="robo-1")
sender.navigate(
    {
        "type": "navigate",
        "position": {"token": "A", "section": "A", "slot": "A01", "color": "blue"},
        "task": "fetch",
    },
    sentence="tới kệ A lấy hộp xanh rồi mang về",
    reply="Đang thực hiện nhiệm vụ.",
)
```

Không cần chờ 20 giây ở phía AIWaiter sau khi gọi `navigate()`: bản sender gửi
bản sao đầu tiên ngay lập tức, sau đó tự gửi thêm hai bản sao cách nhau 20 ms.

## 4. Lệnh điều khiển chuyến đang chạy

Các lệnh này dùng `kind: "control"` và `action.type: "control"`:

```python
sender.control("STOP", sentence="dừng lại")
sender.control("RESUME", sentence="đi tiếp")
sender.control("FORWARD", sentence="đi thẳng")
sender.control("BACKWARD", sentence="đi lùi")
sender.control("LEFT", sentence="quẹo trái")
sender.control("RIGHT", sentence="quẹo phải")
```

JSON tương ứng với `STOP`:

```json
{
  "v": 1,
  "kind": "control",
  "action": {"type": "control", "verb": "STOP"},
  "session": "ca8f31d2",
  "seq": 2,
  "robot_id": "robo-1"
}
```

Ý nghĩa ở AGV:

| Lệnh | Xử lý |
| --- | --- |
| `STOP` | Gửi vận tốc 0 qua `/cmd_vel_keyboard` và tạm dừng mission hiện tại |
| `RESUME` | Tiếp tục đúng mission đang dở, không tạo route mới |
| `FORWARD` | Điều khiển thủ công tiến tới |
| `BACKWARD` | Điều khiển thủ công đi lùi |
| `LEFT` / `RIGHT` | Quay tại chỗ; mặc định 90° |

Muốn truyền góc quay, thêm `angle_deg` (1–360):

```json
{
  "v": 1,
  "kind": "control",
  "action": {"type": "control", "verb": "LEFT", "angle_deg": 45},
  "session": "ca8f31d2",
  "seq": 3
}
```

`STOP` phải được gửi ngay khi người dùng nói dừng; không đưa lệnh này vào hàng
đợi LLM. `RESUME` chỉ có tác dụng với mission đã bị `STOP` tạm dừng.

## 5. ACK và kiểm tra kết nối

AGV trả ACK UDP về đúng địa chỉ/port nguồn của từng datagram:

```json
{"kind":"ack","session":"ca8f31d2","seq":2,"robot_id":"robo-1","v":1}
```

ACK chỉ xác nhận cầu UDP đã nhận gói, không có nghĩa là AGV đã tới kệ. Sender
tự đọc ACK và báo lỗi nếu sau khoảng một giây không nhận được.

Kiểm tra nhanh trên máy AGV:

```bash
ss -lunp | grep 45455
```

Kiểm tra thủ công từ máy AIWaiter (thay IP):

```bash
python3 - <<'PY'
import json, socket

packet = {
    "v": 1,
    "kind": "control",
    "action": {"type": "control", "verb": "STOP"},
    "session": "manual-test",
    "seq": 1,
}
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.settimeout(2)
s.sendto(json.dumps(packet).encode(), ("<IP_MAY_AGV>", 45455))
print("sent")
try:
    print("ack:", s.recvfrom(512)[0].decode())
except socket.timeout:
    print("không nhận ACK — kiểm tra IP, firewall và run_demo.sh")
PY
```

## 6. Để xe bắt đầu chạy nhanh

1. Chạy `./run_demo.sh` một lần và để Gazebo/Nav2/cầu UDP chạy nền trong suốt
   buổi demo.
2. Phía AIWaiter gửi action có cấu trúc (`ROBOT_LINK_SEND_ACTION=1`), không chỉ
   gửi câu nói tự do.
3. Gọi `sender.navigate()` một lần; không đợi ACK rồi mới gửi lại thủ công.
4. Khi tự chạy bằng terminal, dùng:

   ```bash
   ./pick_box.sh --storage A --color blue --deliver
   ```

   Nếu Nav2 đã sẵn sàng, script sẽ hiện `Fast start` và chạy mission ngay.
   Phần kiểm tra readiness chỉ gọi một lần tới ROS action graph để không cộng
   thêm nhiều giây từ các lệnh `ros2` discovery lặp lại.

## 7. Những phần AGV hiện chưa nhận

Để tránh phía AIWaiter tưởng lệnh đã chạy, hiện tại AGV chỉ nhận Storage A và
các task `fetch`/`fetch_hold`. Các token Storage B/C, `goto`, `deliver`,
`cancel`, và `lift` sẽ bị bỏ qua hoặc báo cảnh báo. Có thể mở rộng sau bằng cách
thống nhất thêm action trong cùng giao thức v1.
