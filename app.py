import streamlit as st
from datetime import date, timedelta

# -------------------------
# Utilities
# -------------------------
def clamp_int(x, lo=0, hi=None):
    try:
        x = int(x)
    except Exception:
        x = lo
    if x < lo:
        x = lo
    if hi is not None and x > hi:
        x = hi
    return x

def mm_to_hhmm(m):
    h = m // 60
    mi = m % 60
    return f"{h:02d}:{mi:02d}"

def max_study_in_cap(cap_total, chunk, short_break, long_break, long_every):
    """Max study minutes achievable within cap_total (study+break total), with break rules."""
    remaining = cap_total
    study = 0
    blocks = 0

    while remaining > 0:
        take = chunk if chunk <= remaining else remaining
        study += take
        remaining -= take
        blocks += 1

        if remaining <= 0:
            break

        b_len = long_break if (blocks % long_every == 0) else short_break

        # Only insert a break if there will be time left to study at least 1 minute afterward.
        if remaining >= b_len + 1:
            remaining -= b_len
        else:
            break

    return study

def build_cap_list(today, exam, week_minutes):
    days = (exam - today).days
    caps = []
    for k in range(days):
        d = today + timedelta(days=k)
        caps.append(week_minutes[d.weekday()])
    return caps

def allocate_targets(total_study_capacity, one, weak):
    cycle_total = sum(one)
    full_cycles = total_study_capacity // cycle_total
    remain = total_study_capacity % cycle_total
    extra_percent = (total_study_capacity - full_cycles * cycle_total) * 100 // cycle_total

    target = [one[i] * full_cycles for i in range(len(one))]

    order = list(range(len(one)))
    order.sort(key=lambda i: weak[i], reverse=True)  # 5 -> 1
    for i in order:
        if remain <= 0:
            break
        add = one[i]
        if add > remain:
            add = remain
        target[i] += add
        remain -= add

    return full_cycles, extra_percent, target

def pick_subject(target, weak, max_boost):
    best = -1
    best_score = -1
    for i in range(len(target)):
        if target[i] > 0:
            mult = 1 + (max_boost - 1) * (weak[i] - 1) / 4  # 1.0 ~ max_boost
            score = target[i] * mult
            if score > best_score:
                best_score = score
                best = i
    return best

def plan_day(cap_total, cur_date, exam, names, target, weak, settings):
    """Mutates target. Returns a dict describing the day's plan including a timeline."""
    chunk = settings["chunk"]
    short_break = settings["short_break"]
    long_break = settings["long_break"]
    long_every = settings["long_every"]
    max_boost = settings["max_boost"]

    remaining = cap_total
    study_done = 0
    break_done = 0
    blocks_done = 0
    short_cnt = 0
    long_cnt = 0
    today_alloc = [0] * len(names)
    timeline = []  # (kind, label, mins)

    while remaining > 0:
        if sum(target) <= 0:
            break

        best = pick_subject(target, weak, max_boost)
        if best == -1:
            break

        take = chunk
        if take > target[best]:
            take = target[best]
        if take > remaining:
            take = remaining
        if take <= 0:
            break

        target[best] -= take
        today_alloc[best] += take

        remaining -= take
        study_done += take
        blocks_done += 1
        timeline.append(("공부", names[best], take))

        if remaining <= 0:
            break
        if sum(target) <= 0:
            break

        b_len = long_break if (blocks_done % long_every == 0) else short_break
        if remaining >= b_len + 1:
            remaining -= b_len
            break_done += b_len
            if b_len == long_break:
                long_cnt += 1
                timeline.append(("휴식", "긴 휴식", b_len))
            else:
                short_cnt += 1
                timeline.append(("휴식", "짧은 휴식", b_len))
        else:
            break

    if remaining > 0:
        timeline.append(("여유", "", remaining))

    d_to_exam = (exam - cur_date).days
    return {
        "date": cur_date,
        "d_to_exam": d_to_exam,
        "cap_total": cap_total,
        "study_done": study_done,
        "break_done": break_done,
        "free_left": remaining,
        "short_cnt": short_cnt,
        "long_cnt": long_cnt,
        "today_alloc": today_alloc,
        "timeline": timeline
    }

def compute_plan(today, exam, week_minutes, subjects, settings):
    days = (exam - today).days
    if days <= 0:
        return {"error": "시험 시작일은 오늘보다 뒤여야 합니다."}

    if not subjects:
        return {"error": "과목을 1개 이상 입력하세요."}

    names = [s["name"] for s in subjects]
    one = [int(s["one"]) for s in subjects]
    weak = [int(s["weak"]) for s in subjects]

    cycle_total = sum(one)
    if cycle_total <= 0:
        return {"error": "1회독 총시간 합이 0이면 계획을 만들 수 없습니다."}

    cap_list = build_cap_list(today, exam, week_minutes)
    total_cap = sum(cap_list)

    total_study_capacity = 0
    for cap in cap_list:
        total_study_capacity += max_study_in_cap(
            cap,
            settings["chunk"],
            settings["short_break"],
            settings["long_break"],
            settings["long_every"]
        )

    full_cycles, extra_percent, target = allocate_targets(total_study_capacity, one, weak)

    # Day-by-day plans (mutates target)
    day_plans = []
    target_work = target[:]  # copy
    for i in range(days):
        cur_date = today + timedelta(days=i)
        day_plans.append(plan_day(cap_list[i], cur_date, exam, names, target_work, weak, settings))

    return {
        "summary": {
            "days": days,
            "total_cap": total_cap,
            "total_study_capacity": total_study_capacity,
            "full_cycles": full_cycles,
            "extra_percent": extra_percent
        },
        "names": names,
        "day_plans": day_plans
    }

def timeline_lines(timeline):
    out = []
    tcur = 0
    for kind, label, mins in timeline:
        a = mm_to_hhmm(tcur)
        b = mm_to_hhmm(tcur + mins)
        if kind == "공부":
            out.append(f"{a}~{b}  공부({label}) {mins}분")
        elif kind == "휴식":
            out.append(f"{a}~{b}  {label} {mins}분")
        else:
            out.append(f"{a}~{b}  여유 {mins}분")
        tcur += mins
    return out

def to_csv(day_plans, names):
    lines = ["date,d_to_exam,total_cap,study,break,free,details"]
    for d in day_plans:
        ds = d["date"].isoformat()
        alloc_parts = []
        for i, nm in enumerate(names):
            if d["today_alloc"][i] > 0:
                alloc_parts.append(f"{nm}:{d['today_alloc'][i]}")
        details = " | ".join(alloc_parts)
        lines.append(f"{ds},{d['d_to_exam']},{d['cap_total']},{d['study_done']},{d['break_done']},{d['free_left']},\"{details}\"")
    return "\n".join(lines)

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="시험공부 플래너 (휴식 포함)", layout="wide")
st.title("시험공부 플래너 (휴식 포함 · 타임라인 출력)")

left, right = st.columns([1, 1])

with left:
    today = st.date_input("오늘 날짜", value=date.today())
with right:
    exam = st.date_input("시험 시작일", value=date.today() + timedelta(days=14))

st.divider()
st.subheader("요일별 총 시간 (공부+휴식 포함)")

weekday_labels = ["월", "화", "수", "목", "금", "토", "일"]
cols = st.columns(7)
week_minutes = []
defaults = [240, 240, 240, 240, 240, 180, 120]
for i in range(7):
    with cols[i]:
        week_minutes.append(clamp_int(st.number_input(f"{weekday_labels[i]}", min_value=0, value=defaults[i], step=10)))

st.divider()
st.subheader("설정 (원하면 바꿔도 됨)")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    chunk = clamp_int(st.selectbox("공부 블록(분)", [20, 25, 30, 40, 50], index=2))
with c2:
    short_break = clamp_int(st.selectbox("짧은 휴식(분)", [3, 5, 7, 10], index=1))
with c3:
    long_break = clamp_int(st.selectbox("긴 휴식(분)", [10, 15, 20, 25], index=1))
with c4:
    long_every = clamp_int(st.selectbox("긴 휴식 주기(블록)", [2, 3, 4], index=1), lo=1)
with c5:
    max_boost = float(st.selectbox("약세 우선 최대 배수", [1.0, 1.5, 2.0, 2.5, 3.0], index=2))

settings = {
    "chunk": chunk,
    "short_break": short_break,
    "long_break": long_break,
    "long_every": long_every,
    "max_boost": max_boost
}

st.divider()
st.subheader("과목 입력")
st.caption("표에 입력하세요. 약세(1~5)는 배정에만 반영되고 출력에는 표시되지 않습니다.")

# Initialize default table
if "subjects_table" not in st.session_state:
    st.session_state.subjects_table = [
        {"과목": "수학", "1회독(분)": 600, "약세(1~5)": 4},
        {"과목": "영어", "1회독(분)": 450, "약세(1~5)": 2},
        {"과목": "한국사", "1회독(분)": 300, "약세(1~5)": 3},
    ]

edited = st.data_editor(
    st.session_state.subjects_table,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "과목": st.column_config.TextColumn(required=True),
        "1회독(분)": st.column_config.NumberColumn(min_value=0, step=10, required=True),
        "약세(1~5)": st.column_config.NumberColumn(min_value=1, max_value=5, step=1, required=True),
    }
)
st.session_state.subjects_table = edited

subjects = []
for row in edited:
    name = str(row.get("과목", "")).strip()
    if not name:
        continue
    subjects.append({
        "name": name,
        "one": clamp_int(row.get("1회독(분)", 0), lo=0),
        "weak": clamp_int(row.get("약세(1~5)", 3), lo=1, hi=5)
    })

st.divider()

if st.button("계획 생성", type="primary"):
    plan = compute_plan(today, exam, week_minutes, subjects, settings)

    if "error" in plan:
        st.error(plan["error"])
        st.stop()

    s = plan["summary"]
    st.success(
        f"남은 {s['days']}일 | 총 시간(공부+휴식) {s['total_cap']}분 | "
        f"규칙 적용 후 공부 가능 총량 {s['total_study_capacity']}분 | "
        f"완전 회독 {s['full_cycles']}회 + 추가 {s['extra_percent']}%"
    )

    names = plan["names"]
    day_plans = plan["day_plans"]

    # CSV download
    csv = to_csv(day_plans, names)
    st.download_button("CSV로 다운로드", data=csv, file_name="study_plan.csv", mime="text/csv")

    st.divider()
    st.subheader("날짜별 타임라인")

    for idx, d in enumerate(day_plans):
        if idx % 7 == 0:
            st.markdown(f"### {idx//7 + 1}주차")

        title = f"[{d['date'].strftime('%m/%d')} | D-{d['d_to_exam']}] 총 {d['cap_total']}분 · 공부 {d['study_done']}분 · 휴식 {d['break_done']}분 · 여유 {d['free_left']}분"
        with st.expander(title, expanded=False):
            if d["short_cnt"] + d["long_cnt"] > 0:
                st.write(f"휴식 분배: 짧게 {d['short_cnt']}회, 길게 {d['long_cnt']}회")

            st.write("타임라인")
            st.code("\n".join(timeline_lines(d["timeline"])), language="text")

            # Per-subject totals for the day
            alloc_pairs = []
            for i, nm in enumerate(names):
                if d["today_alloc"][i] > 0:
                    alloc_pairs.append((d["today_alloc"][i], nm))
            alloc_pairs.sort(reverse=True)

            if alloc_pairs:
                st.write("과목 합계")
                for mins, nm in alloc_pairs:
                    st.write(f"- {nm} {mins}분")
