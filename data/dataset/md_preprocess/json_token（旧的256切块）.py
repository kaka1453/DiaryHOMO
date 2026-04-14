import json
import csv
from transformers import AutoTokenizer

# === 核心配置 ===
MAX_TOKENS = 256
OVERLAP_TARGET = 40       # 搜索范围：期望重叠的长度
SAFE_BUFFER = 8          # 安全缓冲
MODEL_ID = "Qwen/Qwen2.5-7B"

# 第一轮切割限制：预留出重叠空间
# 例如：512 - 60 - 10 = 442。保证切出来的块加上重叠后不会爆。
CUT_LIMIT = MAX_TOKENS - OVERLAP_TARGET - SAFE_BUFFER

input_file = "diary_resize.jsonl"
output_file = "diary_resize256.jsonl"
stats_file = "token_stats.csv"

print(f"正在加载 Tokenizer ({MODEL_ID})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

def get_token_len(text):
    """获取精准的 token 数，不带特殊符"""
    return len(tokenizer.encode(text, add_special_tokens=False))

def strict_chunking(text, limit):
    """
    Pass 1: 严格基于 \n\n 的分块策略
    返回: list of segments (strings)
    """
    # 1. 先按 \n\n 彻底打散
    # 注意：split后 \n\n 消失了，我们在重组时只用 \n\n 连接
    raw_blocks = text.split('\n\n')
    
    segments = []
    current_batch = []
    current_tokens = 0
    
    for block in raw_blocks:
        # 去掉首尾空白，避免空块
        if not block.strip():
            continue
            
        block_len = get_token_len(block)
        
        # A. 单个块就是巨无霸，直接超过限制
        if block_len > limit:
            # 1. 先把手里积攒的存了
            if current_batch:
                segments.append("\n\n".join(current_batch))
                current_batch = []
                current_tokens = 0
            
            # 2. 这个巨无霸块单独存，并打标（因为无法按 \n\n 切分了）
            # 用户要求：不能暴力截断，只能标记
            segments.append(f"【BLOCK_TOO_LONG_Check_This】{block}")
            continue
            
        # B. 加上这个块会超限 -> 结算当前批次，开新批次
        # 预估连接符 \n\n 约 1-2 token
        if current_tokens + block_len + 2 > limit:
            if current_batch:
                segments.append("\n\n".join(current_batch))
            current_batch = [block]
            current_tokens = block_len
        
        # C. 还没满 -> 加入当前批次
        else:
            current_batch.append(block)
            current_tokens += (block_len + 2) # +2 是因为未来join时会有\n\n
            
    # 处理最后的尾巴
    if current_batch:
        segments.append("\n\n".join(current_batch))
        
    return segments

def get_strict_overlap(prev_text, target_len):
    """
    Pass 2: 严格的 Overlap 提取
    从 prev_text 的末尾提取一段文本，要求必须是语义完整的。
    """
    if not prev_text:
        return ""

    # 1. 划定搜索范围：取末尾 target_len * 2 的字符 (大致范围)
    # 我们多取一点，防止 token 估算误差
    search_window_char = prev_text[- (target_len * 5):]
    if len(search_window_char) < len(prev_text):
        base_text = search_window_char
    else:
        base_text = prev_text

    # 2. Token 校验：确保 base_text 至少包含 target_len
    # 如果上一段本身就很短（比如只有20token），那就全拿
    total_tokens = tokenizer.encode(base_text, add_special_tokens=False)
    if len(total_tokens) <= target_len:
        return base_text

    # 我们只需要最后 target_len 附近的 token
    # 但我们不能直接按 token 切，必须按字符语义找
    
    # 3. 优先级查找
    # 我们希望找到重叠部分大约是 target_len，所以我们要找的“切入点”
    # 应该是倒数第 target_len 个 token 附近往前的第一个 \n\n
    
    # 策略：在 base_text 中寻找所有可能的分割点
    # 优先级：\n\n (最强) > 。！？ (次优) > \n (勉强)
    # 严禁：逗号、其他
    
    valid_cut_point = -1
    found_type = "None"
    
    # 寻找所有的 \n\n
    # rfind 从右往左找，但我们需要的是“包含最长上下文且不超过 target”？
    # 不，用户要求：“寻找其中的 \n\n，以最长包含的 \n\n 进行截断”
    # 这意味着：在允许的长度范围内（比如50-80 token），尽可能让 Overlap 包含更多的完整段落
    
    # 简单化：我们把 base_text 里的 \n\n 全部找出来
    # 只要从那个 \n\n 开始到结尾的长度 <= MAX_Overlap (比如100) 且 >= MIN_Overlap (比如20)
    
    limit_token_idx = len(total_tokens) - target_len - 20 # 允许往前多看一点
    if limit_token_idx < 0: limit_token_idx = 0
    
    # 将 token 映射回文本稍微麻烦，我们用字符近似法
    # 我们倒序遍历字符
    
    # 优先级 1: \n\n
    # 我们找一段，使得长度适中。
    possible_nn = base_text.rfind('\n\n')
    if possible_nn != -1:
        # 检查剩下的长度
        candidate = base_text[possible_nn:].strip() # 去掉开头的 \n\n
        c_len = get_token_len(candidate)
        # 如果长度在合理范围内 (比如 10 ~ 80)
        if 5 <= c_len <= (target_len + 30):
            return candidate

    # 优先级 2: 句号 (。！？)
    # 如果没找到合适的 \n\n，找句末
    for punc in ["。", "！", "？"]:
        last_punc = base_text.rfind(punc)
        if last_punc != -1:
            candidate = base_text[last_punc+1:] # 不带标点？或者带？通常 overlap 从句首开始
            # 这是一个问题：last_punc 是上一句的结尾。overlap 应该是下一句的开头。
            # 所以应该是 last_punc + 1
            if get_token_len(candidate) <= (target_len + 30):
                return candidate
    
    # 优先级 3: 单换行 \n
    last_n = base_text.rfind('\n')
    if last_n != -1:
        candidate = base_text[last_n:].strip()
        if 5 <= get_token_len(candidate) <= (target_len + 30):
            return candidate

    # 悲剧：找不到任何合规的分割点
    # 比如结尾是 "......今（Target线）天我和小明......" 且没有标点
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

            # === Pass 1: 严格切割 ===
            # 得到的是基于 \n\n 完美分割的片段，或者标记了超长的片段
            segments = strict_chunking(raw_text, CUT_LIMIT)
            
            for i, seg in enumerate(segments):
                # 检查是否是超长标记块
                if "【BLOCK_TOO_LONG" in seg:
                    final_output.append(seg) # 直接保存标记
                    manual_check_blocks += 1
                    continue
                
                final_text = seg
                
                # === Pass 2: 严格拼接 ===
                # 除了第一段，都需要去上一段借 Overlap
                if i > 0:
                    prev_seg = segments[i-1]
                    # 如果上一段是坏块，没法借，跳过 Overlap
                    if "【BLOCK_TOO_LONG" in prev_seg:
                        pass 
                    else:
                        overlap_content = get_strict_overlap(prev_seg, OVERLAP_TARGET)
                        
                        if overlap_content == "【NEED_MANUAL_OVERLAP】":
                            # 拼接失败标记，放在头部
                            final_text = f"【MANUAL_OVERLAP_FIX】{final_text}"
                            overlap_fails += 1
                        elif overlap_content:
                            # 成功拼接：Overlap + \n\n + 当前段
                            # 注意：Overlap 内部可能自带换行，但为了保险，连接处补 \n\n
                            # 除非 Overlap 已经是整段（以 \n\n 开头），我们在 get_strict_overlap 里已经 strip 掉了
                            final_text = f"{overlap_content}\n\n{final_text}"
                            total_overlaps += 1
                
                # === Pass 3: 清洗 ===
                # 再次强调：只去掉最终结果开头的 \n\n
                final_text = final_text.lstrip('\n')
                
                if final_text.strip():
                    final_output.append(final_text)
                    total_chunks += 1

    # 写出文件
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for text in final_output:
            out_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            
    # 打印简报
    print("=" * 40)
    print(f"处理完成！")
    print(f"生成的有效片段总数: {total_chunks}")
    print(f"成功执行的 Overlap: {total_overlaps}")
    print(f"超长需人工处理段落: {manual_check_blocks}")
    print(f"Overlap 失败需检查: {overlap_fails}")
    print(f"输出文件: {output_file}")

def run_stats():
    """最后进行全量 Token 扫描，生成 CSV 报告"""
    print("\n正在生成最终校验报告...")
    stats_data = []
    
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            text = json.loads(line)["text"]
            t_len = get_token_len(text)
            
            status = "PASS"
            if "【" in text: # 检查是否有失败标记
                status = "⚠️ 包含人工标记"
            elif t_len > MAX_TOKENS:
                status = "❌ 超出 512"
            
            # 截取头部预览
            preview = text[:20].replace('\n', '\\n') + "..."
            stats_data.append([i+1, t_len, status, preview])

    with open(stats_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["行号", "Token数", "状态", "头部预览"])
        writer.writerows(stats_data)
    print(f"报告已生成: {stats_file}")

if __name__ == "__main__":
    process_diary()
    run_stats()