import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from autoresearch_purchase_propensity import research


class BudgetTests(unittest.TestCase):
    def test_exhausted_count_or_time_prevents_another_experiment(self):
        with tempfile.TemporaryDirectory() as folder:
            config, state = Path(folder)/'config.json', Path(folder)/'state.json'
            config.write_text(json.dumps({'output_dir': str(Path(folder)/'runs')}))
            for completed, started in [(20, research.time.time()), (0, 0)]:
                state.write_text(json.dumps({'completed': completed, 'started': started,
                                            'max_experiments': 20, 'budget_minutes': 60}))
                with patch.object(research, 'CONFIG', config), patch.object(research, 'STATE', state):
                    with self.assertRaisesRegex(ValueError, 'budget exhausted'):
                        research.prepare()
                self.assertFalse((Path(folder)/'runs').exists())


if __name__ == '__main__':
    unittest.main()
