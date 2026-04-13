import json
import csv
from transformers import AutoTokenizer

# === 核心配置 ===
MAX_TOKENS = 256
OVERLAP_TARGET = 40       
SAFE_BUFFER = 15          # 增加一点缓冲，用于容纳“「裁剪」”标记的长度
MODEL_ID = "Qwen/Qwen2.5-7B"

CUT_LIMIT = MAX_TOKENS - OVERLAP_TARGET - SAFE_BUFFER

input_file = "diary_boa.jsonl"
output_file = "diary_resize.jsonl"
stats_file = "token_stats.csv"

# 裁剪标记
CUT_MARKER = "「裁剪」"

print(f"正在加载 Tokenizer ({MODEL_ID})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

def get_token_len(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

def strict_chunking(text, limit):
    """
    Pass 1: 严格基于 \n\n 的分块策略，并添加裁剪标记
    """
    raw_blocks = text.split('\n\n')
    segments = []
    current_batch = []
    current_tokens = 0
    
    for block in raw_blocks:
        if not block.strip(): continue
        block_len = get_token_len(block)
        
        # A. 单个块超长
        if block_len > limit:
            if current_batch:
                # 给前一个片段加裁剪标记
                segments.append("\n\n".join(current_batch) + CUT_MARKER)
                current_batch = []
                current_tokens = 0
            segments.append(f"【BLOCK_TOO_LONG_Check_This】{block}")
            continue
            
        # B. 加上这个块会超限 -> 结算并添加「裁剪」标记
        if current_tokens + block_len + 2 > limit:
            if current_batch:
                # 只有被切断的部分才加标记
                segments.append("\n\n".join(current_batch) + CUT_MARKER)
            current_batch = [block]
            current_tokens = block_len
        else:
            current_batch.append(block)
            current_tokens += (block_len + 2)
            
    # 处理最后的尾巴：最后一段不加「裁剪」标记，代表自然结束
    if current_batch:
        segments.append("\n\n".join(current_batch))
        
    return segments

def get_strict_overlap(prev_text, target_len):
    """
    Pass 2: 严格的 Overlap 提取 (跳过末尾的「裁剪」标记进行提取)
    """
    if not prev_text: return ""
    
    # 清理掉上一段末尾可能存在的裁剪标记，避免 Overlap 把标记也带入下一段开头
    clean_prev = prev_text.replace(CUT_MARKER, "").strip()
    
    search_window_char = clean_prev[- (target_len * 5):]
    base_text = search_window_char if len(search_window_char) < len(clean_prev) else clean_prev
    total_tokens = tokenizer.encode(base_text, add_special_tokens=False)
    
    if len(total_tokens) <= target_len:
        return base_text

    # 优先级查找逻辑
    possible_nn = base_text.rfind('\n\n')
    if possible_nn != -1:
        candidate = base_text[possible_nn:].strip()
        if 5 <= get_token_len(candidate) <= (target_len + 30):
            return candidate

    for punc in ["。", "！", "？"]:
        last_punc = base_text.rfind(punc)
        if last_punc != -1:
            candidate = base_text[last_punc+1:]
            if get_token_len(candidate) <= (target_len + 30):
                return candidate
    
    last_n = base_text.rfind('\n')
    if last_n != -1:
        candidate = base_text[last_n:].strip()
        if 5 <= get_token_len(candidate) <= (target_len + 30):
            return candidate

    return "【NEED_MANUAL_OVERLAP】"

def process_diary():
    total_chunks = 0
    total_overlaps = 0
    manual_check_blocks = 0
    overlap_fails = 0
    final_output = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if not line.strip(): continue
            data = json.loads(line)
            raw_text = data.get("text", "")

            # 1. 切割 + 标记
            segments = strict_chunking(raw_text, CUT_LIMIT)
            
            for i, seg in enumerate(segments):
                if "【BLOCK_TOO_LONG" in seg:
                    final_output.append(seg)
                    manual_check_blocks += 1
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
                            final_text = f"【MANUAL_OVERLAP_FIX】{final_text}"
                            overlap_fails += 1
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
            
    print("=" * 40)
    print(f"处理完成！")
    print(f"生成的有效片段总数: {total_chunks}")
    print(f"成功执行的 Overlap: {total_overlaps}")
    print(f"带有「裁剪」标记的片段数: {total_chunks - 1 if total_chunks > 1 else 0}")
    print(f"超长需人工处理段落: {manual_check_blocks}")
    print(f"Overlap 失败需检查: {overlap_fails}")

def run_stats():
    stats_data = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            text = json.loads(line)["text"]
            t_len = get_token_len(text)
            status = "PASS"
            if "【" in text: status = "⚠️ 人工标记"
            elif t_len > MAX_TOKENS: status = "❌ 超出"
            stats_data.append([i+1, t_len, status, text[:20].replace('\n', '\\n') + "..."])

    with open(stats_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["行号", "Token数", "状态", "头部预览"])
        writer.writerows(stats_data)

if __name__ == "__main__":
    process_diary()
    run_stats()