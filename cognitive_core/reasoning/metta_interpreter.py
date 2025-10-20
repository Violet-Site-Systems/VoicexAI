"""
Minimal MeTTa-like interpreter for EPPN

This module implements a tiny subset of MeTTa sufficient for expressing
simple facts and rules and running forward-chaining inference. It's not a
full MeTTa implementation — only a safe, small evaluator to enable ethical
reasoning integration in the prototype.

Supported constructs (text format):
- fact: (fact predicate arg1 arg2 ...)
- rule: (=> (head ...) (body1 ...) (body2 ...))  [if all body_i facts hold, head is derived]

Example program:
  (fact person alice)
  (fact has_vehicle alice car)
  (=> (risk alice) (fact person alice) (fact has_vehicle alice car))

The interpreter returns a dict with derived facts and a trace of inferences.
"""

from typing import List, Tuple, Dict, Set
import re


TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")


def tokenize(s: str) -> List[str]:
    return TOKEN_RE.findall(s)


def parse(tokens: List[str], pos: int = 0):
    if pos >= len(tokens):
        raise ValueError("Unexpected end of tokens")
    tok = tokens[pos]
    if tok == '(':
        lst = []
        pos += 1
        while pos < len(tokens) and tokens[pos] != ')':
            elem, pos = parse(tokens, pos)
            lst.append(elem)
        if pos >= len(tokens) or tokens[pos] != ')':
            raise ValueError("Missing closing parenthesis")
        return lst, pos + 1
    elif tok == ')':
        raise ValueError("Unexpected ')'")
    else:
        return tok, pos + 1


def parse_program(s: str) -> List:
    tokens = tokenize(s)
    pos = 0
    ast = []
    while pos < len(tokens):
        node, pos = parse(tokens, pos)
        ast.append(node)
    return ast


def is_fact(node) -> bool:
    return isinstance(node, list) and len(node) >= 1 and node[0] != '=>'


def is_rule(node) -> bool:
    return isinstance(node, list) and len(node) >= 3 and node[0] == '=>'


def fact_to_tuple(fact_node) -> Tuple[str, Tuple[str, ...]]:
    head = fact_node[0]
    args = tuple(str(a) for a in fact_node[1:])
    return head, args


def match_fact(fact: Tuple[str, Tuple[str, ...]], pattern: Tuple[str, Tuple[str, ...]]) -> bool:
    # Simple exact matching (no variables); can be extended later
    return fact == pattern


def evaluate(program: str) -> Dict[str, object]:
    """Evaluate a small MeTTa program and return derived facts and trace.

    Returns:
      {"facts": set of fact tuples, "derived": list of derived fact tuples, "trace": list of steps}
    """
    ast = parse_program(program)
    base_facts: Set[Tuple[str, Tuple[str, ...]]] = set()
    rules: List[Tuple[Tuple[str, Tuple[str, ...]], List[Tuple[str, Tuple[str, ...]]]]] = []

    for node in ast:
        if is_fact(node):
            base_facts.add(fact_to_tuple(node))
        elif is_rule(node):
            # rule format: (=> head body1 body2 ...)
            if len(node) < 3:
                continue
            head = fact_to_tuple(node[1])
            bodies = [fact_to_tuple(b) for b in node[2:]]
            rules.append((head, bodies))

    derived = set()
    trace = []

    # Simple forward chaining: iterate until no new facts
    changed = True
    iteration = 0
    while changed and iteration < 100:
        changed = False
        iteration += 1
        for head, bodies in rules:
            if head in base_facts or head in derived:
                continue
            if all(b in base_facts or b in derived for b in bodies):
                derived.add(head)
                trace.append({
                    "derived": head,
                    "by_rule": {"head": head, "bodies": bodies}
                })
                changed = True

    all_facts = base_facts | derived
    return {
        "facts": all_facts,
        "derived": derived,
        "trace": trace,
    }


if __name__ == "__main__":
    prog = """
    (person alice)
    (has_vehicle alice car)
    (=> (risk alice) (person alice) (has_vehicle alice car))
    """
    print(evaluate(prog))
