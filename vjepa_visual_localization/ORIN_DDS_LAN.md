# Jetson Orin nhận ảnh Gazebo qua ROS 2 DDS và dây LAN trực tiếp

Luồng chạy không dùng socket riêng:

```text
Gazebo /camera (laptop)
  -> worker latest-frame 4 FPS
  -> ROS 2 DDS /vjepa/camera/image_raw
  -> dây LAN
  -> V-JEPA trên Orin
  -> /vjepa_pose + /vjepa_latent + /vjepa_localization/debug qua DDS
  -> dashboard trên laptop
```

Topic ảnh dùng `sensor_msgs/msg/Image`, giữ nguyên timestamp mô phỏng và QoS
`BEST_EFFORT`, `VOLATILE`, `KEEP_LAST(1)`. Khi Orin hoặc mạng chậm, relay bỏ
frame cũ và chỉ truyền frame mới nhất; không tạo hàng đợi làm tăng delay.

## 1. IP tĩnh cho dây LAN nối trực tiếp

Đặt card Ethernet của laptop là `192.168.50.1/24` và card Ethernet của Orin là
`192.168.50.2/24`. Kết nối này không cần gateway hoặc DNS. Có thể đặt bằng giao
diện Network Settings của Ubuntu; chọn IPv4 `Manual` trên đúng card có dây.

Kiểm tra hai chiều:

```bash
# Trên laptop
ping 192.168.50.2

# Trên Orin
ping 192.168.50.1
```

## 2. Môi trường DDS giống nhau trên cả hai máy

Chạy trong mỗi terminal ROS 2 ở cả laptop và Orin:

```bash
export ROS_DOMAIN_ID=77
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export ROS_LOCALHOST_ONLY=0
```

Hai máy phải dùng cùng `ROS_DOMAIN_ID`. DDS discovery đi trực tiếp trên subnet
Ethernet; không cần ROS master và không cần địa chỉ IP trong code.

## 3. Laptop chạy Gazebo và relay DDS, Orin chạy V-JEPA

Trên laptop, tắt localizer GPU nội bộ nhưng vẫn giữ relay và dashboard:

```bash
cd /home/amightyk05/workshop/warehouse_agv_demo
ROS_DOMAIN_ID=77 WAREHOUSE_VJEPA_LOCALIZER=false ./run_demo.sh
```

Chép repository và `outputs/autonomous_map_dense` sang Orin một lần. Sau đó
chạy trên Orin:

```bash
cd /path/to/vjepa_visual_localization
ROS_DOMAIN_ID=77 ./run_orin_vjepa.sh
```

Latent map là dữ liệu tĩnh nằm trên Orin. Chỉ camera live và kết quả V-JEPA đi
qua DDS trong lúc demo.

## Dữ liệu Orin trả về laptop

| Topic DDS | Kiểu ROS 2 | Nội dung |
| --- | --- | --- |
| `/vjepa_pose` | `geometry_msgs/msg/PoseWithCovarianceStamped` | Tọa độ V-JEPA `(x, y, yaw)` trong frame bản đồ kho và covariance |
| `/vjepa_latent` | `std_msgs/msg/Float32MultiArray` | Embedding query 1024 chiều để chiếu PCA và vẽ latent space |
| `/vjepa_localization/debug` | `std_msgs/msg/String` | JSON gồm timestamp tâm clip, compute host, inference ms, top-k IDs/scores, selected clip, similarity và temporal state |

`/vjepa_pose` là vị trí thị giác tương tự GPS trong hệ tọa độ local của nhà kho,
không phải kinh độ/vĩ độ `NavSatFix`. Gazebo truth vẫn ở laptop và chỉ được
dashboard ghép theo timestamp để tính sai số; truth không đi vào Orin encoder.

Dashboard đã có latent map tĩnh trên laptop, nên Orin chỉ cần trả vector query
1024 chiều và `source_id` trong debug. Không truyền lại toàn bộ latent map qua
DDS ở mỗi frame.

## 4. Kiểm tra DDS

Trên Orin:

```bash
ros2 topic info /vjepa/camera/image_raw -v
ros2 topic hz /vjepa/camera/image_raw
ros2 topic bw /vjepa/camera/image_raw
```

Kết quả mong đợi là publisher từ laptop, QoS `BEST_EFFORT`, và xấp xỉ `4 Hz`.
Trên laptop, kiểm tra kết quả trả về từ Orin:

```bash
ros2 topic echo /vjepa_pose --once
ros2 topic echo /vjepa_latent --once
ros2 topic echo /vjepa_localization/debug --once --full-length
```

Nếu ping được nhưng DDS không discovery, kiểm tra hai máy có cùng domain/RMW,
firewall có chặn multicast UDP hay không, và DDS đang dùng card Ethernet thay
vì một VPN interface.
