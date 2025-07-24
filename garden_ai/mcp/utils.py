import ast

def extract_function_signature(code: str):
    lines = code.splitlines()
    signature_lines = []
    collecting = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def "):
            collecting = True

        if collecting:
            signature_lines.append(line)
            if stripped.endswith(":"):
                break

    return " ".join(signature_lines).strip()


def parse_function_signature(func_dec: str):
    if func_dec.endswith(":"):
        func_dec += "\n\tpass"

    tree = ast.parse(func_dec)

    function_params = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = node.args.args
            defaults = node.args.defaults
            # Fill missing defaults with None
            num_missing_defaults = len(args) - len(defaults)
            defaults = [None] * num_missing_defaults + defaults

            param_list = []
            for arg, default in zip(args, defaults):
                param_info = {
                    "name": arg.arg,
                    "type": ast.unparse(arg.annotation) if arg.annotation else None,
                    "default": ast.unparse(default) if default else None,
                }
                param_list.append(param_info)

            function_params.append(param_list)

    return function_params
