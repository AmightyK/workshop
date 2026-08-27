Hãy sửa trực tiếp **Warehouse Questions UI hiện tại** dựa trên implementation đang có.

Ảnh/UI hiện tại đang hiển thị:

**Warehouse Questions - 10 of 20**

và bên dưới render **10 question cards cùng lúc theo layout 2 cột**.

Tôi muốn GIỮ cách hoạt động **nhiều câu hỏi xuất hiện cùng lúc**, nhưng thay đổi như sau.

## 1. Từ 10 cards xuống CHÍNH XÁC 5 cards

Không làm kiểu:

Q1 → trả lời → Q2 → Q3...

Tôi muốn:

**Q1 + Q2 + Q3 + Q4 + Q5 HIỂN THỊ CÙNG LÚC trên cùng màn hình.**

Tức là thay UI hiện tại:

**10 cards cùng lúc**

thành:

**5 cards cùng lúc.**

Header cũng phải đổi tương ứng, không còn:

`Warehouse Questions - 10 of 20`

hay:

`10 câu hỏi luôn sẵn sàng`

Hãy đổi thành nội dung phù hợp với bộ **5 câu hỏi cuối cùng**.

## 2. ĐỪNG chỉ lấy 5 câu đầu tiên trong 20 câu hiện tại

Đây là yêu cầu quan trọng nhất.

Trước khi quyết định 5 câu nào sẽ xuất hiện, hãy kiểm tra **TOÀN BỘ trajectory thực tế của robot từ Start đến Finish**.

Hãy thực sự đi theo timeline/trajectory và xác định:

* pose của robot;
* orientation/heading;
* robot đang đứng yên hay di chuyển;
* tốc độ;
* robot đang đi thẳng hay đang turning;
* đoạn tiếp theo robot sẽ rẽ trái/phải hay đi thẳng;
* obstacle/landmark thực tế;
* LiDAR/camera information nếu có;
* waypoint/checkpoint tương ứng.

Sau khi hiểu toàn bộ route mới thiết kế **5 câu hỏi tốt nhất**.

Không bắt buộc sử dụng Q01–Q05 hiện tại.

Nếu câu hiện tại sai, mơ hồ hoặc không có đủ dữ liệu để chứng minh đáp án thì **XÓA và viết câu mới**.

## 3. Kiểm tra logic của từng câu theo trajectory

Hiện tại tôi nghi ngờ một số câu hỏi/đáp án không đúng với trạng thái thực tế của robot.

Ví dụ:

Q1 có thể đang tương ứng với một trạng thái robot cụ thể, nhưng Q2 lại kết luận:

`Robot đi thẳng.`

Trong khi trajectory thực tế tại context/frame tương ứng chưa chắc robot đang đi thẳng.

Không được suy luận kiểu này.

Với TỪNG câu trong 5 câu cuối cùng, hãy xác minh:

`question`
→ `source frame / timestamp / waypoint`
→ `robot pose`
→ `orientation`
→ `velocity`
→ `trajectory trước đó`
→ `trajectory tiếp theo`
→ `sensor/environment evidence`
→ `ground truth`

Chỉ khi evidence thực sự chứng minh được đáp án thì mới sử dụng câu hỏi đó.

## 4. 5 câu phải được lấy từ các thời điểm hợp lý trên MỘT full route

Hãy xem toàn bộ route trước rồi chọn 5 thời điểm có ý nghĩa để đặt câu hỏi.

Không cần chia khoảng cách chính xác bằng nhau, nhưng nên đại diện hợp lý cho hành trình:

`START ── Q1 ───── Q2 ─── Q3 ───── Q4 ─── Q5 ── FINISH`

Q1–Q5 phải theo đúng thứ tự thời gian của trajectory.

Không được lấy frame ngẫu nhiên rồi ghép thành 5 câu.

## 5. Viết lại câu hỏi nếu cần

Ưu tiên câu hỏi có ground truth rõ ràng.

Ví dụ có thể hỏi về:

* Robot hiện đang đứng yên hay di chuyển?
* Robot đang đi thẳng hay turning?
* Robot sắp thực hiện maneuver nào?
* Landmark/obstacle nào thực sự xuất hiện trong observation?
* Robot đang tiến gần hay rời xa một landmark?
* Hướng chuyển động hiện tại là gì?
* Không gian phía trước có đủ để robot tiếp tục hay robot đang giảm tốc?

Nhưng đây chỉ là ví dụ.

**Không cố sử dụng một dạng câu hỏi nếu dữ liệu thực tế không hỗ trợ nó.**

Nếu một câu hỏi khác phù hợp với trajectory hơn thì hãy tự tạo câu hỏi khác.

## 6. Đáp án phải chính xác hơn

Hiện tại có các đáp án dạng:

`Robot đi thẳng.`

`Tốc độ đang giữ tương đối đều.`

`Đi tiếp bình thường.`

`Di thẳng và giữ tốc độ hiện tại.`

Không được giữ những đáp án này chỉ vì chúng đã có trong config.

Hãy verify lại bằng dữ liệu thực tế.

Nếu trajectory cho thấy robot chuẩn bị rẽ thì đáp án không được nói "đi thẳng".

Nếu velocity đang thay đổi đáng kể thì không được nói "giữ tốc độ".

Nếu obstacle phía trước không đủ bằng chứng thì không được tự khẳng định có obstacle.

**Ground truth phải đến từ dữ liệu, không phải từ wording cũ.**

## 7. Không show thẳng "ĐÁP ÁN" như UI hiện tại

UI trong ảnh hiện tại đang render:

`ĐÁP ÁN:`
`Robot đang đứng yên.`

ngay trên card.

Cách này làm mất ý nghĩa của question UI.

Hãy chuyển mỗi card thành dạng interactive question.

Ví dụ:

**Q01**

Robot đang đứng yên hay đang di chuyển?

○ Đứng yên
○ Di chuyển chậm
○ Di chuyển nhanh

Người dùng chọn một đáp án.

Làm tương tự cho **cả 5 cards đang hiển thị cùng lúc**.

Sau đó có một nút:

**Submit Answers**

Sau submit mới reveal:

* Correct / Incorrect
* Correct answer
* Score tổng, ví dụ `4 / 5`

## 8. REDESIGN UI — UI hiện tại quá thô

Không chỉ đổi text/config.

Hãy redesign trực tiếp OpenCV question panel hiện tại.

UI hiện tại có các vấn đề:

* cards quá giống bảng debug;
* màu nền nâu/xám nặng;
* hierarchy yếu;
* khoảng cách text chưa tốt;
* một số text bị chen/chồng;
* label `ĐÁP ÁN` gây rối;
* header chiếm diện tích nhưng không cung cấp nhiều thông tin;
* question number nhỏ và khó scan;
* card chưa có selected/hover/correct/incorrect state rõ ràng.

Hãy làm UI theo phong cách:

**Modern Robotics / AI Research Dashboard**

Ưu tiên clean, dark, professional.

5 cards phải được bố trí đẹp trong không gian hiện tại.

 layout khác nếu hợp lý hơn.

Không bắt buộc giữ layout hiện tại nếu có phương án đẹp hơn.

Mỗi card nên có:

* Q number rõ ràng;
* question text nổi bật;
* answer choices;
* selected state;
* đủ padding;
* alignment chính xác;
* text wrapping chuẩn;
* không overlap;
* consistent spacing;
* subtle border;
* hover/active feedback nếu OpenCV UI hiện tại hỗ trợ.

## 9. Không được chỉ sửa bằng cảm tính — chạy FULL ROUTE

Sau khi implement xong:

**Hãy chạy simulation từ Start đến Finish.**

Quan sát toàn bộ trajectory.

Sau đó kiểm tra lại từng Q1–Q5 dựa trên route thực tế.

Tạo một bảng kiểm tra nội bộ:

| Question | Frame/Waypoint | Robot State | Evidence | Ground Truth |
| -------- | -------------- | ----------- | -------- | ------------ |
| Q1       | ...            | ...         | ...      | ...          |
| Q2       | ...            | ...         | ...      | ...          |
| Q3       | ...            | ...         | ...      | ...          |
| Q4       | ...            | ...         | ...      | ...          |
| Q5       | ...            | ...         | ...      | ...          |

Nếu bất kỳ câu nào không chứng minh được ground truth từ dữ liệu thực tế, **thay câu hỏi đó**.

## 10. Acceptance criteria

Task chỉ được coi là hoàn thành khi:

* Có **chính xác 5 câu hỏi**.
* **Cả 5 câu hiển thị cùng lúc**, giống concept UI hiện tại nhưng chỉ còn 5.
* Mỗi câu có answer choices để người dùng chọn.
* Không show ground truth trước Submit.
* Có Submit Answers.
* Có score/result sau Submit.
* Q1–Q5 theo đúng thứ tự của một full trajectory.
* Tất cả ground truth đã được verify bằng dữ liệu thực tế.
* Không còn câu hỏi mâu thuẫn với robot pose/movement/trajectory.
* Text không overlap.
* UI được redesign đẹp và chuyên nghiệp hơn đáng kể so với UI hiện tại.
* Đã chạy full route Start → Finish để kiểm chứng.

**Đừng chỉ sửa số `10` thành `5`. Đừng chỉ lấy Q01–Q05 hiện tại. Đừng chỉ redesign UI. Phải kiểm tra lại toàn bộ trajectory, chọn lại 5 câu hỏi hợp lý và xác minh lại từng ground-truth answer trước khi hoàn thành.**
