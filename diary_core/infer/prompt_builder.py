from __future__ import annotations

from diary_core.infer.prompt_contract import DiaryContract


DEFAULT_SYSTEM_PROMPT = (
    "你是一个私人日记写作助手。你的任务是根据用户给出的写作契约生成一篇私人日记。\n"
    "必须围绕 MAIN_TOPIC 和 TOPIC_TERMS 写作，不得迁移到写作契约列出的旧主题禁区；FORBIDDEN_DRIFT_TOPICS 是硬禁区，不是参考建议。\n"
    "正文只能展开 SCENE_BEATS 里的本次事件推进，不要接着旧日记时间线续写，也不要补课堂、作业、图书馆、股票、算法等旧经历。\n"
    "写作风格要像真实私人日记：具体、口语、碎碎念，遇到搞笑/荒诞事件要保留梗词和现场感，可以有“蚌、难蚌、寄、邪恶、离谱”等作者口癖。\n"
    "如果用户输入很短，只能围绕输入本身写当下感受和模糊日常，不要回顾过去具体经历，不要扩写成旧人物长故事、学习计划、股票交易或技术文档。\n"
    "只能使用简体中文；不要日语假名、繁体字、无意义英文长串、乱码、Markdown 表格、代码块或链接。\n"
    "只输出单篇日记正文，不要解释，不要标题。"
)


OLD_MEMORY_FORBIDDEN_TERMS = [
    "浩海钓具",
    "钓鱼用品店",
    "反向伞",
    "购物",
    "过去的聊天",
    "聊天记录",
    "顾问",
    "数学课",
    "预习",
    "明天还得",
    "线上上课",
    "线下课",
    "看番",
    "教室改造",
    "圣地",
    "分校区",
    "图书馆",
    "高数",
    "线性代数",
    "高中作业",
    "晚自习",
    "昨天晚上群里那人",
    "睡大觉",
    "语文作文",
    "历史暑假书",
    "剪纸",
    "大明瑾",
    "人物档案",
]


def build_messages(contract: DiaryContract, attachments: dict | None = None, config: dict | None = None) -> list[dict]:
    config = config or {}
    attachments = attachments or {}
    system_prompt = config.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
    user_content = "\n".join(
        [
            "[RAW_PROMPT]",
            contract.raw_prompt.strip(),
            "",
            "[MAIN_TOPIC]",
            contract.main_topic,
            "",
            "[TOPIC_TERMS]",
            _format_list(getattr(contract, "topic_terms", [])),
            "",
            "[MUST_INCLUDE_TERMS]",
            _format_list(_must_include_terms(contract)),
            "",
            "[SCENE_BEATS]",
            _format_list(_scene_beats(contract)),
            "",
            "[CURRENT_TOPIC_LOCK]",
            contract.topic_lock,
            "",
            "[ALLOWED_FACTS]",
            _format_list(contract.allowed_facts),
            "",
            "[STYLE_HINTS]",
            _format_list(contract.style_hints),
            "",
            "[FORBIDDEN_DRIFT_TOPICS]",
            _format_list(select_prompt_forbidden_topics(contract)),
            "",
            "[FORMAT_FORBIDDEN]",
            _format_list(getattr(contract, "format_forbidden", [])),
            "",
            "[STYLE_EXECUTION]",
            _style_execution_rule(contract),
            "",
            "[ANTI_GENERIC_RULE]",
            _anti_generic_rule(contract),
            "",
            "[SELF_CHECK_BEFORE_OUTPUT]",
            _self_check_rule(contract),
            "",
            "[DRIFT_AVOIDANCE_RULE]",
            _drift_avoidance_rule(contract),
            "",
            "[OLD_MEMORY_BAN]",
            _old_memory_ban_rule(contract),
            "",
            "[FACT_BOUNDARY_HINT]",
            _fact_boundary_hint(contract),
            "",
            "[ATTACHMENTS]",
            _format_attachments(attachments),
            "",
            "[OUTPUT]",
            "只输出单篇连续日记正文。不要输出标题、解释、列表、分点、表格、代码块、标签或链接。",
        ]
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def render_prompt(tokenizer, messages: list[dict], use_chat_template: bool = True) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return _fallback_render_messages(messages)


def _format_list(items: list[str]) -> str:
    if not items:
        return "- 无"
    return "\n".join(f"- {item}" for item in items)


def _format_attachments(attachments: dict) -> str:
    if not attachments or not attachments.get("items"):
        return "- 无附件。本轮不要引入额外事实。"
    lines = []
    for item in attachments["items"]:
        lines.append(f"- {item}")
    return "\n".join(lines)


def select_prompt_forbidden_topics(contract: DiaryContract, max_items: int = 24) -> list[str]:
    topic_text = _compact(" ".join([contract.raw_prompt, contract.main_topic, " ".join(getattr(contract, "topic_terms", []))]))
    groups = [
        ("股票/期权/投资类旧主题", ["股票", "期权", "投资", "TradingLog", "上证", "杠杆", "交易"]),
        ("算法/代码/技术文档类旧主题", ["算法", "代码", "技术文档", "HTTP", "SELECT", "布尔矩阵"]),
        ("考试/复习/课程类旧主题", ["考试", "复习", "课程", "作业", "高数", "考研", "NUS", "雅思"]),
        ("实验室/论文/项目/报告类旧主题", ["实验室", "论文", "项目", "报告", "开题"]),
        ("旧聊天记录/旧人物长故事", ["旧聊天记录", "微信聊天", "截图", "旧人物长故事", "旧群聊复盘"]),
        ("设备购物/Apple/闲鱼类旧主题", ["AirPods", "苹果店", "Apple Store", "AppleWatch", "闲鱼", "屏幕"]),
        ("游戏媒体/B站/Genshin/YouTube 类旧主题", ["Genshin", "原神", "B站", "bilibili", "YouTube", "GitHub", "番剧"]),
        ("外部链接/Markdown表格/代码块", ["外部链接", "Markdown 表格", "代码块", "HTML 标签"]),
        ("日语假名/繁体字/无意义英文长串/乱码", ["日语假名", "繁体字", "英文长串", "乱码字符", "混杂外语叙述"]),
    ]
    selected = []
    for label, keywords in groups:
        if any(_contains_compact(topic_text, keyword) for keyword in keywords):
            continue
        selected.append(label)

    explicit = [
        "股票",
        "期权",
        "算法",
        "考试",
        "NUS",
        "实验室",
        "高数",
        "线性代数",
        "数学课",
        "预习",
        "作业",
        "图书馆",
        "晚自习",
        "旧聊天记录",
        "聊天记录",
        "顾问",
        "外部链接",
        "Markdown 表格",
        "日语假名",
        "英文长串",
    ]
    for item in explicit:
        if len(selected) >= max_items:
            break
        if _contains_compact(topic_text, item):
            continue
        if item in getattr(contract, "forbidden_drift_topics", []) and item not in selected:
            selected.append(item)
    return selected[:max_items]


def _fact_boundary_hint(contract: DiaryContract) -> str:
    fact_types = getattr(contract, "forbidden_fact_types", [])
    weak_hint = "缺少细节时，优先写感受和模糊日常，不要为了丰富内容迁移旧日记事件。"
    if not fact_types:
        return weak_hint
    return f"{weak_hint} 弱提醒：不要主动添加{_join_inline(fact_types[:4])}等未授权具体事实。"


def _style_execution_rule(contract: DiaryContract) -> str:
    styles = set(getattr(contract, "style_hints", []))
    risk_tags = set(getattr(contract, "risk_tags", []))
    humor_styles = {"搞笑", "幽默", "吐槽", "荒诞", "离谱", "难蚌", "戏剧化", "自嘲", "反差", "段子", "夸张", "破防", "场景化"}
    if styles & humor_styles or "humor_or_absurd_prompt" in risk_tags:
        return "\n".join(
            [
                "本篇按“作者风味搞笑日记”写，不要写成普通道理反思。",
                "正文写 2-3 个自然段即可；第一段进入 SCENE_BEATS 第1步，第二段写尴尬/反差，第3段用内心 OS 或自嘲收住。",
                "保留 RAW_PROMPT 中最有梗的词和物件，比如难蚌、寄、破防、白嫖、邪恶、屁股垫、分手厨房、被迫围观等，不要改成抽象概念。",
                "用具体动作、现场画面、对话感、内心弹幕和反差比喻制造笑点；至少写一句“我当时脑子里的吐槽/OS”。",
                "允许出现少量作者口癖：蚌、难蚌、寄、邪恶、离谱、属实，但不要为了口癖脱离本次事件。",
                "可以轻微夸张，像真实碎碎念一样吐槽，但每个笑点都要回到 TOPIC_TERMS。",
                "结尾用一个小落点或自嘲收住，例如“属实难蚌”“今日评价：寄但有节目效果”，不要升华成大道理。",
            ]
        )
    if getattr(contract, "length_hint", "") == "short":
        return "短 prompt 要短写，只围绕输入本身写当下感受和一点碎碎念，不编旧经历。"
    return "普通日常要写自然、具体、有一点碎碎念；多写现场动作和内心 OS，不要写成总结报告。"


def _anti_generic_rule(contract: DiaryContract) -> str:
    return "\n".join(
        [
            "不要把日记写成“事件总结 + 人生道理”。少用：这让我意识到、总的来说、这种体验让我感到、也许这就是生活、这是值得我学习的。",
            "不要写“朋友总是能带来帮助和支持”“这是很有价值的体验”“生活中的小确幸”“迎接明天”这类泛泛总结；把抽象评价换成一个动作、一句对话或一个内心弹幕。",
            "不要输出括号里的写作建议，例如“这里可以加一些……”；日记里不能露出提示词或写作说明。",
            "不要机械复述 RAW_PROMPT 作为第一句；要改写成日记开场，直接进入现场。",
            "不要只总结情绪；要写具体场景、动作、对话感、内心 OS 和一个小笑点。",
        ]
    )


def _drift_avoidance_rule(contract: DiaryContract) -> str:
    lines = [
        "FORBIDDEN_DRIFT_TOPICS 中的内容不得出现在正文里；如果想写到禁区词，必须立刻改写回 MAIN_TOPIC。",
        "每一段至少要能对应一个 TOPIC_TERMS，不能先写主题、后面转到学习、股票、技术、旧聊天、设备购物、游戏媒体或链接。",
        "不要写“先是/下午/现在/刚才”串联一堆无关日程；本篇只写 SCENE_BEATS 给出的这一件事。",
        "不要用【回忆录】、情况1/情况2、编号分点、任务清单或资料摘录式结构；正文必须像一篇连续的私人日记。",
    ]
    if getattr(contract, "length_hint", "") == "short":
        lines.append("本条输入很短，只写 1-2 个自然段；不要补过去经历、家庭旧事、小区邻居、学校课程、考试复习、明天安排等具体故事。")
    return "\n".join(lines)


def _self_check_rule(contract: DiaryContract) -> str:
    terms = _join_inline(getattr(contract, "topic_terms", [])[:6]) or contract.main_topic
    must_terms = _join_inline(_must_include_terms(contract))
    return "\n".join(
        [
            f"输出前自检：每一段是否都能对应这些锚点：{terms}。",
            f"正文必须自然出现这些本题核心词：{must_terms or '本次主题词'}；不要把它们替换成其他旧场景。",
            "如果某句话出现课堂、作业、数学课、预习、图书馆、股票、算法、旧聊天、明天安排等未授权旧内容，删掉那句话。",
            "如果结尾开始升华人生道理，改成一句普通碎碎念或自嘲。",
        ]
    )


def _old_memory_ban_rule(contract: DiaryContract) -> str:
    topic_text = _compact(" ".join([contract.raw_prompt, contract.main_topic, " ".join(getattr(contract, "topic_terms", []))]))
    forbidden = [term for term in OLD_MEMORY_FORBIDDEN_TERMS if not _contains_compact(topic_text, term)]
    if not forbidden:
        return "本题已授权相关词时可使用；其余仍不得扩写成旧日记时间线。"
    return (
        "下面这些是训练集中容易冒出来的旧记忆，不是本次素材，正文不要出现："
        f"{_join_inline(forbidden[:20])}。"
        "如果缺细节，宁可围绕 RAW_PROMPT 写短一点，也不要借旧流水账补字数。"
    )


def _must_include_terms(contract: DiaryContract) -> list[str]:
    terms = [term for term in getattr(contract, "topic_terms", []) if 1 <= len(term) <= 12]
    if not terms:
        return []
    return terms[:4]


def _scene_beats(contract: DiaryContract) -> list[str]:
    prompt = contract.raw_prompt.strip()
    terms = [term for term in getattr(contract, "topic_terms", []) if term]
    if getattr(contract, "length_hint", "") == "short":
        if _contains_compact(prompt, "朋友") and _contains_compact(prompt, "吃饭"):
            return [
                "开场：我和朋友坐下来吃饭，直接写饭桌现场。",
                "发展：写聊天、夹菜、笑出来的小瞬间，别补餐厅名和旧经历。",
                "收尾：用一句“今天确实挺开心”的碎碎念结束，不要转到数学课、预习或明天安排。",
            ]
        return [
            f"只围绕“{contract.main_topic}”写当下感受，不补旧经历。",
            f"可用锚点：{_join_inline(terms[:5]) or '本次输入本身'}。",
            "收尾用一句轻微碎碎念，不做大道理总结。",
        ]

    humor = "humor_or_absurd_prompt" in set(getattr(contract, "risk_tags", []))
    clauses = _prompt_clauses(prompt)
    if humor:
        if _contains_compact(prompt, "分手厨房") and _contains_compact(prompt, "妹妹"):
            return [
                "起因：我躲在书房假装刻苦工作，其实是在躲客厅那个脾气火爆的妹妹。",
                "冲突：她可能抓我去玩“自杀式”的分手厨房，我脑内警报直接拉满。",
                "收尾：写我的内心 OS 和难蚌自嘲，不要写剪纸、作业、旧人物或其他旧事件。",
            ]
        if _contains_compact(prompt, "红烧肉") and _contains_compact(prompt, "葱油饼"):
            return [
                "起因：山东深夜在宿舍开火做红烧肉和葱油饼。",
                "冲突：醋香扩散，全寝室被迫围观，空气像被调成了酸菜鱼模式。",
                "收尾：写室友反应和我的内心 OS，不要转到学习、课程或旧聊天。",
            ]
        if _contains_compact(prompt, "屁股垫") and _contains_compact(prompt, "抱枕"):
            return [
                "起因：我搬着德川/纯平双面抱枕进电梯，场面已经有点不妙。",
                "冲突：女生问这是不是“屁股垫”，我当场难蚌，脑子开始加载失败。",
                "收尾：写一个尴尬回应和自嘲落点，不要转到图书馆、高数、作业或旧日程。",
            ]
        if _contains_compact(prompt, "恋爱脑"):
            return [
                "起因：今天我恋爱脑又发作了，语气要无奈又有点可爱。",
                "冲突：围绕一个小触发点写胡思乱想，比如看见消息、照片或一句话就开始脑补。",
                "收尾：用“我又开始了”式自嘲收住，不要写过去的聊天记录、顾问、数学课、番、复习或明天安排。",
            ]
        return [
            f"起因：只写 RAW_PROMPT 里的本次场景：{clauses[0] if clauses else contract.main_topic}。",
            f"冲突/笑点：围绕这些梗词展开：{_join_inline(terms[:6]) or contract.main_topic}。",
            "收尾：用一句内心 OS 或自嘲收住，不升华、不扩写旧经历。",
        ]

    if clauses:
        if _contains_compact(prompt, "朋友") and _contains_compact(prompt, "吃饭"):
            return [
                "开场：第一句必须含有“朋友”和“吃饭”，直接写饭桌现场，不能改成逛店、钓具、购物或其他旧经历。",
                "发展：写聊天、夹菜、笑出来的小瞬间，别补餐厅名和旧经历。",
                "收尾：用一句“今天确实挺开心”的碎碎念结束，不要写友谊价值、小确幸、数学课、预习或明天安排。",
            ]
        beats = [f"开场：围绕“{clauses[0]}”直接进入现场。"]
        if len(clauses) > 1:
            beats.append(f"发展：只补“{clauses[1]}”相关的动作、对话或感受。")
        else:
            beats.append(f"发展：围绕主题词 {_join_inline(terms[:5]) or contract.main_topic} 写具体场景。")
        beats.append("收尾：用一句自然的碎碎念结束，不写人生总结。")
        return beats

    return [
        f"开场：围绕“{contract.main_topic}”直接进入现场。",
        f"发展：只使用这些锚点：{_join_inline(terms[:6]) or '本次输入'}。",
        "收尾：短落点，不升华，不补旧经历。",
    ]


def _join_inline(items: list[str]) -> str:
    return "、".join(items)


def _prompt_clauses(prompt: str) -> list[str]:
    cleaned = prompt.strip(" \n\t。.!！?？")
    cleaned = cleaned.replace("写一篇", "").replace("日记", "")
    parts = []
    for part in cleaned.replace("，", ",").replace("。", ",").replace("；", ",").split(","):
        item = part.strip(" “”，,。.!！?？")
        if item and len(item) <= 42:
            parts.append(item)
    return parts[:3]


def _compact(text: str) -> str:
    return "".join(str(text).split()).lower()


def _contains_compact(text: str, keyword: str) -> bool:
    return _compact(keyword) in text


def _fallback_render_messages(messages: list[dict]) -> str:
    chunks = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        chunks.append(f"[{role}]\n{content}")
    chunks.append("[ASSISTANT]\n")
    return "\n\n".join(chunks)
