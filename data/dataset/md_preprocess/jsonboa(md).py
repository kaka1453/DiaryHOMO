import re
import json

input_file = "diary_boa.md"
output_file = "diary_boa.jsonl"

# ========= 1️⃣ 严格定义「分隔符行」 =========

TITLE_PATTERNS = [
    # # 文件: xxx.md（允许乱码，任意非换行字符）
    re.compile(r'^\s*#\s*文件\s*:\s*[^\n]+?\.md\s*$', re.IGNORECASE),

    # # Diary for xxx
    re.compile(r'^\s*#\s*Diary\s+for\s+.+$', re.IGNORECASE),

    # ## / ### + 严格日期（不能带“计划”“Timetable”等）
    re.compile(
        r'^\s*#{2,3}\s*'
        r'\d{4}[-/.]\d{1,2}[-/.]\d{1,2}'
        r'(?:\s*周[一二三四五六日天])?\s*$'
    ),
]

def is_separator_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    return any(p.match(s) for p in TITLE_PATTERNS)

# ========= 2️⃣ 找到所有「分隔符块」 =========

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

separator_blocks = []   # [(start_line, end_line)]
i = 0
n = len(lines)

while i < n:
    if is_separator_line(lines[i]):
        start = i
        j = i
        # 合并连续的分隔符行
        while j + 1 < n and is_separator_line(lines[j + 1]):
            j += 1
        # 顺便吃掉后面的空行
        k = j + 1
        while k < n and lines[k].strip() == "":
            k += 1
        separator_blocks.append((start, k))
        i = k
    else:
        i += 1

# ========= 3️⃣ 打印所有匹配到的分隔符（用于检查） =========

print("===== 🧭 匹配到的分隔符块 =====")
for idx, (s, e) in enumerate(separator_blocks, 1):
    print(f"\n--- 分隔符块 {idx} ---")
    for line in lines[s:e]:
        print(line.rstrip())

print("\n===== 🧭 分隔符检查完毕 =====\n")

# ========= 4️⃣ 提取正文并写入 JSONL =========

entries = []

for idx, (s, e) in enumerate(separator_blocks):
    content_start = e
    content_end = separator_blocks[idx + 1][0] if idx + 1 < len(separator_blocks) else n
    text = "".join(lines[content_start:content_end]).strip()

    if text:
        entries.append({"text": text})

with open(output_file, "w", encoding="utf-8") as f:
    for obj in entries:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"✅ 已生成 JSONL：{output_file}，共 {len(entries)} 条")
