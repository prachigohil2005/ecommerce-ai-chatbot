import re

def test_extraction(query_clean):
    comparison_targets = []
    # Match "compare <target1> with/to/and <target2>"
    match = re.search(r"compare\s+(.+?)\s+(?:with|to|and)\s+(.+)", query_clean)
    if match:
        comparison_targets = [match.group(1).strip(), match.group(2).strip()]
    return comparison_targets

def test_resolution(comparison_targets, history):
    resolved_targets = []
    for target in comparison_targets:
        if str(target).lower() in ["this", "it", "previous option", "that option", "the recommended product", "the previous one"]:
            prev_product = None
            for turn in reversed(history):
                if turn["role"] == "assistant":
                    # Extract product title inside **...**
                    matches = re.findall(r"\*\*(.*?)\*\*", turn["content"])
                    if matches:
                        prev_product = matches[0]
                        break
            if prev_product:
                resolved_targets.append(prev_product)
            else:
                resolved_targets.append(target)
        else:
            resolved_targets.append(target)
    return resolved_targets

# Test data from screenshot
query = "compare this with dell inspiron"
history = [
    {"role": "user", "content": "suggest laptop"},
    {"role": "assistant", "content": "Hello! I am your virtual sales associate today. I found some excellent options for your request. Let me recommend the **Acer Inspiron 15 Laptop (Intel Core i5 12th Gen, 32 GB RAM, 1 TB SSD)**.\n\n**Why I recommend this:**\n- Reliable option..."}
]

print("Query:", query)
extracted = test_extraction(query.lower())
print("Extracted targets:", extracted)
resolved = test_resolution(extracted, history)
print("Resolved targets:", resolved)
