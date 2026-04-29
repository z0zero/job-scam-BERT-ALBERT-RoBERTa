import re

def check_red_flags(text):
    """
    Check for heuristic red flags in the job description.
    Returns a list of warning strings.
    """
    flags = []
    text_lower = text.lower()
    
    # 1. Bank/Payment requests
    if re.search(r'\b(bank account|credit card|wire transfer|western union|send money|upfront payment)\b', text_lower):
        flags.append("Contains requests for bank details or upfront payments.")
        
    # 2. Instant/Guaranteed Income
    if re.search(r'\b(guaranteed income|easy money|get rich quick|earn \$.* per week|no experience required.*earn)\b', text_lower):
        flags.append("Promises instant, guaranteed, or unusually high income for little effort.")
        
    # 3. Suspicious Urgency
    if re.search(r'\b(act now|immediate hire|urgent hire|apply immediately)\b', text_lower):
        flags.append("Uses suspicious urgency to pressure applicants.")
        
    # 4. Excessive CAPSLOCK
    words = text.split()
    caps_words = [w for w in words if w.isupper() and len(w) > 2]
    if len(words) > 10 and (len(caps_words) / len(words)) > 0.3:
        flags.append("Uses an excessive amount of ALL CAPS (unprofessional).")
        
    # 5. Generic Salutations
    if re.search(r'\b(dear applicant|dear candidate|to whom it may concern)\b', text_lower):
        flags.append("Uses generic salutations instead of a professional greeting.")
        
    return flags
