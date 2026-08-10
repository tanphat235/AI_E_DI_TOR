"""The desktop UI (Phase 9).

:mod:`.tasks` is pure Python and holds everything worth testing; :mod:`.workers`,
:mod:`.models` and :mod:`.window` are the Qt layer over it. Nothing is imported from here at
package level, because importing ``app.ui`` must not require PySide6 to be installed.
"""

__all__: list[str] = []
