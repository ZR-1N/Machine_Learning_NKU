"""Windows 无 LibreOffice 时，复用技能渲染器栅格化 Word 导出的 PDF。"""
from pathlib import Path
import importlib.util
import os

root=Path(__file__).resolve().parent
skill=Path.home()/'.codex/plugins/cache/openai-primary-runtime/documents/26.1022.11436/skills/documents/render_docx.py'
runtime=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies'
poppler=next((runtime/'native/poppler').rglob('pdftoppm.exe')).parent
os.environ['PATH']=str(poppler)+os.pathsep+os.environ['PATH']
spec=importlib.util.spec_from_file_location('canonical_docx_renderer',skill)
renderer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
def existing_word_pdf(*args, **kwargs):
    return str(root/'report_qa/report.pdf'), 'Native Microsoft Word PDF export; LibreOffice unavailable on this Windows runtime.'
renderer.convert_to_pdf=existing_word_pdf
renderer.rasterize(str(root/'实验2回归分析实验报告_已完成.docx'),str(root/'report_qa'),120,True,False)
