import src.agents.quality_eval as q
from src.agents.quality_eval import (run_rule_checks, evaluate_quality, route_after_quality,
                                     LLMJudgeVerdict, JudgeItem)

SOURCES_RAW = [
    {"title": "트위니 회사소개서", "url": ""},                                   # URL 없는 로컬 문서
    {"title": "로봇신문 트위니 기사", "url": "https://www.irobotnews.com/news/1"},
    {"title": "KIRIA 물류로봇 시장 보고서", "url": "https://kiria.org/report"},
    {"title": "마스오토 뉴스", "url": "https://marsauto.com/news"},
]

def good_report():
    return """# 보고서
## SUMMARY
- 트위니는 자율주행 물류로봇을 상용화해 공급 중이다 [1][2]
- 마스오토는 카메라 기반 화물 자율주행을 실증하고 있다 [4]
## 기업별 평가
### 트위니
#### 기술 성숙도
- 특허를 다수 보유했고 양산 단계에 들어섰다고 회사 측 자료는 밝힌다 [1]
#### 시장성
- 물류로봇 시장 규모는 연평균 성장률 두 자릿수로 전망된다 [3]
#### 이해관계자
- 주요 고객사로 제조 대기업과 협력하고 있다고 보도되었다 [2]
#### 도메인 적용
- 공장 내 물류 현장 적용 사례가 보도되었다 [2]
#### 리스크
- 매출 대부분이 소수 고객사에 의존하는 한계가 있다 [2]
### 마스오토
#### 기술 성숙도
- 카메라 기반 E2E 모델로 상용화 이전 실증 단계다 [4]
#### 시장성
- 화물 자율주행 시장 규모는 성장 초기 단계로 평가된다 [3]
#### 이해관계자
- 물류 운송사 파트너와 협력해 운행 데이터를 쌓고 있다 [4]
#### 도메인 적용
- 고속도로 화물 운송 실증 사례가 확인된다 [4]
#### 리스크
- 레벨4 규제가 정비되지 않아 상용화 시점이 불확실하다 [3]
## REFERENCES
[1] 트위니 회사소개서
[2] 로봇신문 트위니 기사 - https://irobotnews.com/news/1/
[3] KIRIA 물류로봇 시장 보고서 - http://kiria.org/report
[4] 마스오토 뉴스 - https://marsauto.com/news
"""

STATE = {"trace_id": "t1", "sources": [], "evaluations": [
    {"startup": "트위니", "sources": SOURCES_RAW[:3]},
    {"startup": "마스오토", "sources": SOURCES_RAW[2:]},
]}

def rules(report):
    return run_rule_checks(report, q.collect_sources(STATE), q.collect_companies(STATE))

def test_good_report_passes_all_rules():
    r = rules(good_report())
    assert all(not v for v in r.values()), r

def test_no_inline_citation_fails_groundedness():
    rep = q._CITE_RE.sub("", good_report().split("## REFERENCES")[0]) + "## REFERENCES" + good_report().split("## REFERENCES")[1]
    assert rules(rep)["groundedness"]

def test_unknown_reference_fails():
    rep = good_report().replace("[4] 마스오토 뉴스 - https://marsauto.com/news", "[4] 지어낸 블로그 - https://fake.example.com")
    assert any("찾을 수 없는" in v for v in rules(rep)["groundedness"])

def test_recommendation_fails_neutrality_but_reference_title_ignored():
    rep = good_report().replace("## REFERENCES", "- 트위니를 최우수 기업으로 추천합니다 [1]\n## REFERENCES")
    assert rules(rep)["neutrality"]
    rep2 = good_report().replace("[3] KIRIA 물류로봇 시장 보고서", "[3] KIRIA 물류로봇 시장 보고서 압도적 성장")
    assert not rules(rep2)["neutrality"]
    rep3 = good_report().replace("주요 고객사로", "최고기술책임자 출신 창업자와 주요 고객사로")
    assert not rules(rep3)["neutrality"]
    rep4 = good_report().replace("## 기업별 평가", "## 투자 추천 기업 상세")
    assert rules(rep4)["neutrality"]

def test_coverage_per_company():
    rep = good_report().replace("#### 도메인 적용\n- 고속도로 화물 운송 실증 사례가 확인된다 [4]\n", "")
    r = rules(rep)["coverage"]
    assert r and "마스오토" in r[0], r

def test_bias_risk_per_company_and_concentration():
    rep = good_report().replace("- 레벨4 규제가 정비되지 않아 상용화 시점이 불확실하다 [3]", "- 운행 거리를 늘려가고 있다 [4]")
    rep = rep.replace("#### 리스크\n- 운행", "#### 기타\n- 운행")
    assert any("마스오토" in v for v in rules(rep)["bias_control"]), rules(rep)["bias_control"]
    one = q._CITE_RE.sub("[2]", good_report().split("## REFERENCES")[0]) + "## REFERENCES" + good_report().split("## REFERENCES")[1]
    assert rules(one)["bias_control"]

def _judge(scores):
    def fake(state, items):
        return LLMJudgeVerdict(items=[JudgeItem(item=i, score=scores.get(i, 5), reason="r") for i in items], feedback="fb")
    return fake

def test_node_pass_and_fail_and_route(monkeypatch, tmp_path):
    monkeypatch.setattr(q, "LOG_DIR", tmp_path)
    monkeypatch.setattr(q, "run_llm_judge", _judge({}))
    st = {**STATE, "report": good_report(), "report_revision_count": 0}
    out = evaluate_quality(st)
    assert out["eval_result"].verdict == "PASS" and route_after_quality({**st, **out}) == "end"
    assert (tmp_path / "t1.jsonl").exists()

    monkeypatch.setattr(q, "run_llm_judge", _judge({"neutrality": 3}))
    out = evaluate_quality(st)
    assert out["eval_result"].verdict == "FAIL" and not out["eval_result"].neutrality_pass
    assert "neutrality" not in out["eval_result"].feedback or "중립성" in out["eval_result"].feedback
    assert route_after_quality({**st, **out}) == "reporter"

def test_judge_error_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(q, "LOG_DIR", tmp_path)
    def boom(state, items): raise RuntimeError("api down")
    monkeypatch.setattr(q, "run_llm_judge", boom)
    out = evaluate_quality({**STATE, "report": good_report()})
    assert out["eval_result"].verdict == "FAIL" and out["last_error"] and out["node_status"]["quality_eval"] == "FAIL"

def test_rule_fail_skips_llm_only_for_that_item(monkeypatch, tmp_path):
    monkeypatch.setattr(q, "LOG_DIR", tmp_path)
    seen = {}
    def spy(state, items):
        seen["items"] = items
        return LLMJudgeVerdict(items=[JudgeItem(item=i, score=5, reason="r") for i in items])
    monkeypatch.setattr(q, "run_llm_judge", spy)
    rep = good_report().replace("## REFERENCES", "- 트위니를 추천합니다 [1]\n## REFERENCES")
    out = evaluate_quality({**STATE, "report": rep})
    assert "neutrality" not in seen["items"] and "groundedness" in seen["items"]
    assert out["eval_result"].groundedness_pass and not out["eval_result"].neutrality_pass

def test_revision_cap():
    from src.state import QualityEvalResult
    fail = QualityEvalResult(groundedness_pass=False, neutrality_pass=False,
                             bias_control_pass=False, verdict="FAIL", feedback="")
    # 미달 1~3회차 → 재작성, 4회차 → 종료 (재작성 최대 3회)
    assert [route_after_quality({"eval_result": fail, "report_revision_count": n}) for n in (1, 2, 3, 4)] == \
           ["reporter", "reporter", "reporter", "end"]

def test_prompt_build():
    p = q._build_prompt({**STATE, "report": "R"}, ["groundedness", "coverage"])
    assert "- groundedness" in p and "- coverage" in p and "{report}" not in p and p.rstrip().endswith("R")

def test_coverage_keyword_fallback_without_subheadings():
    rep = """## SUMMARY
- 요약 문장입니다 충분히 길게 씁니다 [1]
### 트위니
- 특허를 보유하고 시장 규모가 크며 고객사와 협력하고 도입 사례가 있다 [1]
### 마스오토
- 특허를 보유하고 시장 규모가 크며 파트너와 협력한다 [4]
## REFERENCE
[1] 트위니 회사소개서
[4] 마스오토 뉴스 - https://marsauto.com/news
"""
    r = rules(rep)["coverage"]
    assert len(r) == 1 and "마스오토" in r[0] and "도메인 적용" in r[0], r
