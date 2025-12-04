def create_dummy_model(device):
    """Create a dummy calculator for testing."""
    import numpy as np
    from ase.calculators.calculator import Calculator, all_changes

    class DummyCalc(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(
            self, atoms=None, properties=["energy"], system_changes=all_changes
        ):
            super().calculate(atoms, properties, system_changes)
            self.results["energy"] = -1.0 * len(self.atoms)
            self.results["forces"] = np.zeros((len(self.atoms), 3))
            self.results["stress"] = np.zeros(6)

    return DummyCalc()
