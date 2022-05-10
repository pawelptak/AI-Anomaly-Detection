
def get_tokens_for_tfidf(input):
    tokens_by_slash = str(input.encode('utf-8')).split('/')
    tokens_by_slash += str(input.encode('utf-8')).split('\\')
    all_tokens = []
    for i in tokens_by_slash:
        tokens = str(i).split('-')
        tokens_by_dot = []
        for j in range(len(tokens)):
            temp_tokens = str(tokens[j]).split('.')
            tokens_by_dot += temp_tokens
        all_tokens += tokens + tokens_by_dot
    all_tokens += tokens_by_slash
    all_tokens = list(set(all_tokens))
    if 'com' in all_tokens:
        all_tokens.remove('com')
    if 'pl' in all_tokens:
        all_tokens.remove('pl')
    return all_tokens