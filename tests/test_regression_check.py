from tools.regression_check import CompareResult, compare_values


def test_compare_values_accepts_subset_with_float_tolerance() -> None:
    result = CompareResult()

    compare_values(
        actual={"status": "success", "score": 0.901, "items": [{"name": "person", "box": [1.0, 2.0]}]},
        expected={"status": "success", "score": 0.9, "items": [{"name": "person"}]},
        result=result,
        tolerance=0.01,
    )

    assert result.ok


def test_compare_values_reports_missing_key() -> None:
    result = CompareResult()

    compare_values(actual={"status": "success"}, expected={"missing": True}, result=result)

    assert not result.ok
    assert "$.missing: missing key" in result.errors
