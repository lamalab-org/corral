from corral.sandbox._state import generate_state_restore_code, generate_state_save_code


class TestStateCode:
    def test_restore_code_compiles(self):
        code = generate_state_restore_code("/tmp/state.pkl")
        compile(code, "<restore>", "exec")

    def test_save_code_compiles(self):
        code = generate_state_save_code("/tmp/state.pkl")
        compile(code, "<save>", "exec")

    def test_restore_code_includes_path(self):
        code = generate_state_restore_code("/tmp/test_state.pkl")
        assert "/tmp/test_state.pkl" in code

    def test_save_code_includes_path(self):
        code = generate_state_save_code("/tmp/test_state.pkl")
        assert "/tmp/test_state.pkl" in code
