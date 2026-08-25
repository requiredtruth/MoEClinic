"""PySide6 MoE routing diagnostics control panel."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def demo()->dict:
 p=subprocess.run([sys.executable,"-m","moeclinic",str(ROOT/"examples/collapsed.jsonl"),"--json"],cwd=ROOT,text=True,capture_output=True)
 return {"status":"PASS" if p.returncode==0 else "FAIL","scenario":"bundled synthetic collapsed routing trace","returncode":p.returncode,"report":p.stdout,"errors":p.stderr}
def launch()->int:
 from PySide6.QtCore import QProcess
 from PySide6.QtWidgets import QApplication,QFileDialog,QHBoxLayout,QLabel,QMainWindow,QPlainTextEdit,QProgressBar,QPushButton,QVBoxLayout,QWidget
 class W(QMainWindow):
  def __init__(self):
   super().__init__();self.setWindowTitle("MoEClinic Control Panel");self.resize(920,620);self.p=QProcess(self);r=QWidget();l=QVBoxLayout(r);l.addWidget(QLabel("Mixture-of-experts routing health diagnostics"))
   row=QHBoxLayout()
   for label,path in [("Healthy Demo",ROOT/"examples/healthy.jsonl"),("Collapsed Demo",ROOT/"examples/collapsed.jsonl")]:
    b=QPushButton(label);b.clicked.connect(lambda checked=False,p=path:self.start(str(p)));row.addWidget(b)
   pick=QPushButton("Analyze JSONL…");pick.clicked.connect(self.pick);stop=QPushButton("Stop");stop.clicked.connect(self.p.kill);row.addWidget(pick);row.addWidget(stop);l.addLayout(row)
   self.bar=QProgressBar();l.addWidget(self.bar);self.status=QLabel("Ready");l.addWidget(self.status);self.out=QPlainTextEdit();self.out.setReadOnly(True);l.addWidget(self.out);self.setCentralWidget(r)
   self.p.readyReadStandardOutput.connect(self.read);self.p.readyReadStandardError.connect(self.readerr);self.p.finished.connect(self.finish)
  def pick(self):p,_=QFileDialog.getOpenFileName(self,"Select routing trace","","JSONL (*.jsonl)");self.start(p) if p else None
  def start(self,path):
   if self.p.state()!=QProcess.NotRunning:return
   self.out.clear();self.status.setText("Analyzing");self.bar.setRange(0,0);self.p.setWorkingDirectory(str(ROOT));self.p.start(sys.executable,["-m","moeclinic",path,"--json"])
  def read(self):self.out.appendPlainText(bytes(self.p.readAllStandardOutput()).decode(errors="replace").rstrip())
  def readerr(self):self.out.appendPlainText(bytes(self.p.readAllStandardError()).decode(errors="replace").rstrip())
  def finish(self,c,_):self.bar.setRange(0,100);self.bar.setValue(100 if c==0 else 0);self.status.setText("Complete" if c==0 else f"Failed ({c})")
 app=QApplication([]);w=W();w.show();return app.exec()
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--demo",action="store_true");a=p.parse_args()
 if a.demo:
  r=demo();print(json.dumps(r,indent=2));return 0 if r["status"]=="PASS" else 1
 return launch()
if __name__=="__main__":raise SystemExit(main())
