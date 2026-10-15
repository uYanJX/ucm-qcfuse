"""Separate generated thinking and answer tokens before scoring a response."""


class OutputProgress:
    """Observe first content timestamps at engine-output granularity."""

    def __init__(self, tokenizer, thinking):
        self.control = set(tokenizer.all_special_ids)
        self.start = tokenizer.convert_tokens_to_ids("<think>")
        self.end = tokenizer.convert_tokens_to_ids("</think>")
        self.control.update((self.start, self.end))
        self.thinking = thinking
        self.in_thinking = thinking
        self.seen = 0
        self.first_thinking_time = None
        self.first_answer_time = None

    def update(self, ids, now):
        for token in ids[self.seen :]:
            if self.thinking and token == self.end:
                self.in_thinking = False
            if self.thinking and token == self.start:
                self.in_thinking = True
            if token in self.control:
                continue
            if self.in_thinking and self.first_thinking_time is None:
                self.first_thinking_time = now
            if not self.in_thinking and self.first_answer_time is None:
                self.first_answer_time = now
        self.seen = len(ids)


def split_output(token_ids, tokenizer, thinking):
    ids = list(token_ids)
    control = set(tokenizer.all_special_ids)
    start_id = tokenizer.convert_tokens_to_ids("<think>")
    end_id = tokenizer.convert_tokens_to_ids("</think>")
    control.update((start_id, end_id))
    if thinking:
        ends = [index for index, token in enumerate(ids) if token == end_id]
        if ends:
            boundary = ends[-1]
            starts = [
                index for index, token in enumerate(ids[:boundary]) if token == start_id
            ]
            start = starts[0] + 1 if starts else 0
            thought_ids, answer_ids = ids[start:boundary], ids[boundary + 1 :]
            status = "complete"
        else:
            # An unfinished reasoning stream must not be treated as an answer,
            # even if it happens to mention a ground-truth word or option.
            thought_ids, answer_ids = ids, []
            status = "unfinished"
    else:
        thought_ids, answer_ids = [], ids
        status = "disabled"
    thought_content = [token for token in thought_ids if token not in control]
    answer_content = [token for token in answer_ids if token not in control]
    return {
        "thinking_status": status,
        "thinking_tokens": len(thought_content),
        "answer_tokens": len(answer_content),
        "control_tokens": len(ids) - len(thought_content) - len(answer_content),
        "total_output_tokens": len(ids),
        "thinking_text": tokenizer.decode(
            thought_content, skip_special_tokens=True
        ).strip(),
        "answer_text": tokenizer.decode(
            answer_content, skip_special_tokens=True
        ).strip(),
    }
