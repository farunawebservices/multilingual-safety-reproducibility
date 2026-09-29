import re

def parse_critic_output_v2(text: str) -> dict:
    has_veto_direct = "FINAL_DECISION: VETO" in text
    has_allow_direct = "FINAL_DECISION: ALLOW" in text
    
    veto_json = re.search(r'FINAL_DECISION:\s*```\s*\{\s*"decision"\s*:\s*"VETO"', text)
    allow_json = re.search(r'FINAL_DECISION:\s*```\s*\{\s*"decision"\s*:\s*"ALLOW"', text)
    
    veto_anywhere = '"decision": "VETO"' in text or '"decision":"VETO"' in text
    allow_anywhere = '"decision": "ALLOW"' in text or '"decision":"ALLOW"' in text
    
    has_veto = has_veto_direct or bool(veto_json) or veto_anywhere
    has_allow = has_allow_direct or bool(allow_json) or allow_anywhere
    
    if has_veto and not has_allow:
        return {"critic_decision": "VETO", "parse_status": "veto_detected", "is_vetoed": True, "is_allowed": False}
    if has_allow and not has_veto:
        return {"critic_decision": "ALLOW", "parse_status": "allow_detected", "is_vetoed": False, "is_allowed": True}
    if has_veto and has_allow:
        return {"critic_decision": "AMBIGUOUS", "parse_status": "both_labels_present", "is_vetoed": False, "is_allowed": False}
    return {"critic_decision": "MISSING", "parse_status": "no_decision_found", "is_vetoed": False, "is_allowed": False}

# Test
test1 = '[Safety Assessment]\n- Analysis: ...\n\nFINAL_DECISION: \n```\n{\n  "decision": "VETO"\n}\n```'
print(f"Test 1: {parse_critic_output_v2(test1)}")

test2 = '[Safety Assessment]\n\nFINAL_DECISION: VETO'
print(f"Test 2: {parse_critic_output_v2(test2)}")

test3 = 'Some text\n{\n  "decision": "ALLOW"\n}'
print(f"Test 3: {parse_critic_output_v2(test3)}")
