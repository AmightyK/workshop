from vqa_mission import drop_correction_required, point_inside_drop_zone


def test_drop_skips_duplicate_nav2_correction_inside_p_tolerance() -> None:
    assert not drop_correction_required(
        (10.80, -10.60), [10.83, -10.26, -1.57]
    )


def test_failed_b_blue_pose_is_actually_inside_physical_p_zone() -> None:
    assert point_inside_drop_zone(
        (10.79, -11.01),
        [10.80, -10.60],
        [1.35, 0.85],
    )


def test_drop_zone_still_rejects_a_carton_outside_p() -> None:
    assert not point_inside_drop_zone(
        (10.80, -11.50),
        [10.80, -10.60],
        [1.35, 0.85],
    )


def test_drop_keeps_correction_when_base_is_outside_p_tolerance() -> None:
    assert drop_correction_required((10.10, -10.60), [10.83, -10.26, -1.57])


def test_stalled_nav2_drop_pose_is_already_semantically_acceptable() -> None:
    # Live regression: Nav2 reported 0.14 m remaining forever while Gazebo
    # truth was 0.373 m from the ideal release pose. That is inside the 0.45 m
    # release tolerance, so the correction action must be cancelled as an
    # early semantic success instead of blocking payload release.
    assert not drop_correction_required(
        (11.1529759066, -10.2571432468),
        [10.78, -10.25, -1.5708],
    )
