import json
import csv
from transformers import AutoTokenizer

# === 核心配置 ===
MAX_TOKENS = 256
OVERLAP_TARGET = 30       
MIN_OVERLAP_TOKENS = 10   # 严防偷渡门槛
MAX_OVERLAP_LIMIT = 60   
SAFE_BUFFER = 5          # 只需要留一点给 \n\n 的空间即可，不用留给标记了

MODEL_ID = "Qwen/Qwen2.5-7B"

# 第一轮切割限制
CUT_LIMIT = MAX_TOKENS - OVERLAP_TARGET - SAFE_BUFFER

input_file = "diary_resize256.jsonl"
output_file = "256diary_resize.jsonl"
stats_file = "token_stats.csv"
error_log_file = "error_log.txt" # 专门存放报错的文件

print(f"正在加载 Tokenizer ({MODEL_ID})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# 全局错误收集器
issue_report = []

def get_token_len(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

def log_issue(line_num, issue_type, content_snippet):
    """收集错误信息"""
    msg = f"[行 {line_num}] {issue_type}: {content_snippet}"
    issue_report.append(msg)

def strict_chunking(text, limit):
    """
    Pass 1: 严格基于 \n\n 的分块 (无标记版)
    """
    raw_blocks = text.split('\n\n')
    segments = []
    current_batch = []
    current_tokens = 0
    
    for block in raw_blocks:
        if not block.strip(): continue
        block_len = get_token_len(block)
        
        # A. 单个块超长 (无法按 \n\n 切分)
        if block_len > limit:
            if current_batch:
                segments.append("\n\n".join(current_batch))
                current_batch = []
                current_tokens = 0
            
            # 这是一个错误块，必须标记让用户看到
            segments.append(f"【BLOCK_TOO_LONG_Check_This】{block}")
            continue
            
        # B. 加上这个块会超限 -> 结算
        if current_tokens + block_len + 2 > limit:
            if current_batch:
                segments.append("\n\n".join(current_batch))
            current_batch = [block]
            current_tokens = block_len
        else:
            current_batch.append(block)
            current_tokens += (block_len + 2)
            
    if current_batch:
        segments.append("\n\n".join(current_batch))
        
    return segments

def find_all_occurrences(text, sub):
    indices = []
    pos = text.find(sub)
    while pos != -1:
        indices.append(pos)
        pos = text.find(sub, pos + 1)
    return indices

def get_strict_overlap(prev_text, target_len):
    """
    Pass 2: 贪心且有底线的 Overlap 提取
    """
    if not prev_text: return ""
    
    # 这里的 prev_text 已经是干净的文本，没有标记需要 replace
    clean_prev = prev_text.strip()
    
    search_limit_char = target_len * 6
    if len(clean_prev) < search_limit_char:
        search_window = clean_prev
        offset_base = 0
    else:
        search_window = clean_prev[-search_limit_char:]
        offset_base = len(clean_prev) - len(search_window)

    # 长度极短特例检查
    total_tokens = get_token_len(search_window)
    if total_tokens <= target_len + 20: 
        if total_tokens >= MIN_OVERLAP_TOKENS:
            return search_window
        elif len(clean_prev) < 20: 
            return clean_prev 
    
    # === 分层贪心搜索 ===
    def try_extract(cut_indices, offset_adjustment=0):
        for idx in cut_indices:
            candidate = search_window[idx + offset_adjustment:].strip()
            t_len = get_token_len(candidate)
            if MIN_OVERLAP_TOKENS <= t_len <= MAX_OVERLAP_LIMIT:
                return candidate
        return None

    # Priority 1: \n\n
    nn_indices = find_all_occurrences(search_window, '\n\n')
    res = try_extract(nn_indices, offset_adjustment=2)
    if res: return res

    # Priority 2: 句末标点
    punc_indices = []
    for punc in ["。", "！", "？"]:
        punc_indices.extend(find_all_occurrences(search_window, punc))
    punc_indices.sort() 
    res = try_extract(punc_indices, offset_adjustment=1)
    if res: return res

    # Priority 3: 单换行
    n_indices = find_all_occurrences(search_window, '\n')
    res = try_extract(n_indices, offset_adjustment=1)
    if res: return res

    return "【NEED_MANUAL_OVERLAP】"

def process_diary():
    total_chunks = 0
    total_overlaps = 0
    
    final_output = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if not line.strip(): continue
            data = json.loads(line)
            raw_text = data.get("text", "")

            # 1. 切割 (无标记)
            segments = strict_chunking(raw_text, CUT_LIMIT)
            
            for i, seg in enumerate(segments):
                # 检查切割阶段的致命错误
                if "【BLOCK_TOO_LONG" in seg:
                    final_output.append(seg)
                    log_issue(line_idx+1, "❌ 无法切割的超长块", seg[:50].replace('\n', ' ')+"...")
                    continue
                
                final_text = seg
                
                # 2. 拼接
                if i > 0:
                    prev_seg = segments[i-1]
                    if "【BLOCK_TOO_LONG" in prev_seg:
                        pass 
                    else:
                        overlap_content = get_strict_overlap(prev_seg, OVERLAP_TARGET)
                        
                        if overlap_content == "【NEED_MANUAL_OVERLAP】":
                            final_text = f"【MANUAL_OVERLAP_FIX】\n{final_text}"
                            log_issue(line_idx+1, "⚠️ Overlap失败(无合适断点)", prev_seg[-50:].replace('\n', ' ')+"...")
                        elif overlap_content:
                            final_text = f"{overlap_content}\n\n{final_text}"
                            total_overlaps += 1
                
                # 3. 清洗
                final_text = final_text.lstrip('\n')
                if final_text.strip():
                    final_output.append(final_text)
                    total_chunks += 1

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for text in final_output:
            out_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            
    print("处理完成。")
    return total_chunks, total_overlaps

def run_stats():
    stats_data = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            text = json.loads(line)["text"]
            t_len = get_token_len(text)
            status = "PASS"
            if "【" in text: status = "⚠️ 包含错误标记"
            elif t_len > MAX_TOKENS + 20: status = "❌ 超限"
            
            preview = text[:30].replace('\n', '\\n') + "..."
            stats_data.append([i+1, t_len, status, preview])

    with open(stats_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["行号", "Token数", "状态", "预览"])
        writer.writerows(stats_data)

def print_summary(chunks, overlaps):
    """在最后统一展示所有问题"""
    print("\n" + "="*50)
    print("              执行结果汇总              ")
    print("="*50)
    print(f"总生成片段: {chunks}")
    print(f"成功Overlap: {overlaps}")
    
    if not issue_report:
        print("\n✅ 完美！没有发现任何需要人工干预的问题。")
    else:
        print(f"\n🚫 发现 {len(issue_report)} 个问题，请务必处理：")
        print("-" * 50)
        # 写入独立日志文件
        with open(error_log_file, 'w', encoding='utf-8') as f:
            for issue in issue_report:
                print(issue) # 打印到控制台
                f.write(issue + "\n") # 写入文件
        print("-" * 50)
        print(f"详细错误日志已保存至: {error_log_file}")
        print("请打开 diary_resize.jsonl 搜索 '【' 快速定位修复。")

if __name__ == "__main__":
    chunks, overlaps = process_diary()
    run_stats()
    print_summary(chunks, overlaps)