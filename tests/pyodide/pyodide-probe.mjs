// Pyodide runtime probe — validates the in-browser Python environment.
//
// Confirms that:
//   1. Pyodide loads successfully (CPython via Emscripten)
//   2. Standard library modules used by the SPARQL bridge are importable
//   3. micropip (Pyodide's package installer) is functional
//
// Extend this file to load the larql package once it is Pyodide-packaged:
//
//   await pyodide.loadPackage('micropip');
//   await pyodide.runPythonAsync(`
//     import micropip
//     await micropip.install('larql')
//     from larql import LqlQuery
//   `);

import { loadPyodide } from 'pyodide';

async function main() {
  console.log('Loading Pyodide...');
  const pyodide = await loadPyodide();

  const version = pyodide.runPython('import sys; sys.version');
  console.log('Python version:', version);

  // Standard library smoke test — modules used by the SPARQL bridge
  pyodide.runPython(`
import json
import re
import urllib.parse
import collections
import itertools
import functools
print("Standard library imports: OK")
  `);

  // Verify micropip — required for installing the larql wheel into Pyodide
  await pyodide.loadPackage('micropip');
  const micropipVersion = pyodide.runPython(
    'import micropip; micropip.__version__'
  );
  console.log('micropip version:', micropipVersion);

  console.log('Pyodide runtime probe: OK');
}

main().catch((e) => {
  console.error('Pyodide probe failed:', e);
  process.exit(1);
});
