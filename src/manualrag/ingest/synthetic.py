"""Original six-page fictional printer manual for reproducible tests."""
from pathlib import Path
import pymupdf

SECTIONS = ["Setup","Control panel","Error codes","Paper jams","Troubleshooting","Specifications"]

def create_synthetic_manual(path: Path) -> Path:
    """Write an original manual with selectable text, a vector figure, and a ruled table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for i,title in enumerate(SECTIONS):
        p = doc.new_page(width=595,height=842)
        p.insert_text((42,42),"ManualLens Lab | ML-100 fictional printer",fontsize=10,color=(0.2,0.3,0.4))
        p.insert_text((42,85),title,fontsize=22,fontname="hebo")
        p.insert_text((42,810),f"Physical PDF page {i+1} | Synthetic test fixture, not a real product",fontsize=9)
    def text(page: int,y: float,value: str) -> None:
        doc[page].insert_text((42,y),value,fontsize=11)
    text(0,125,"Place the ML-100 on a level surface with 10 cm clearance behind it.")
    text(0,165,"WARNING: Use only the supplied power adapter. Keep the printer dry.")
    text(0,210,"1. Remove the blue shipping tape.\n2. Load up to 50 sheets of A4 plain paper, print side up.\n3. Connect the supplied adapter to the rear power port.\n4. Press Power once. The green Ready light turns on.")
    text(0,310,"For USB setup, connect the cable to the rear USB-B port after power-on.")
    text(0,345,"For a test page, hold Resume for three seconds while Ready is steady green.")
    p=doc[1]
    p.draw_rect((70,150,520,400),color=(0.15,0.3,0.5),fill=(0.94,0.96,0.98),width=2)
    for x,label in [(150,"1 Power"),(295,"2 Resume"),(435,"3 Cancel")]:
        p.draw_circle((x,245),28,color=(0.1,0.4,0.6),width=2)
        p.insert_text((x-26,305),label,fontsize=11)
    p.draw_circle((445,185),7,color=(0.1,0.6,0.2),fill=(0.1,0.6,0.2))
    p.insert_text((350,215),"4 Ready indicator",fontsize=11)
    text(1,445,"Control panel: Power is left; Resume is center; Cancel is right.")
    text(1,475,"The Ready indicator is at the top-right of the control panel.")
    text(1,510,"Steady green Ready means ready to print. Blinking green means printing.")
    text(1,545,"Press Cancel once to stop the current print job.")
    text(1,580,"Press Resume once after reloading paper to continue printing.")
    text(1,615,"The USB-B port and power port are on the rear panel.")
    p=doc[2]
    xs=[42,112,268,553]; ys=[130,166,218,270,322,374]
    for x in xs:p.draw_line((x,ys[0]),(x,ys[-1]),color=(0.2,0.3,0.4))
    for y in ys:p.draw_line((xs[0],y),(xs[-1],y),color=(0.2,0.3,0.4))
    rows=[["Code","Meaning","Remedy"],["E04","Paper jam","Clear the paper jam; see Paper jams."],
          ["F-12","Rear cover open","Close the rear cover until it clicks."],["E 05","Paper tray empty","Load A4 paper, then press Resume."],
          ["E06","Cartridge missing","Insert the cartridge until it clicks."]]
    for r,row in enumerate(rows):
        for c,cell in enumerate(row):
            p.insert_textbox((xs[c]+6,ys[r]+8,xs[c+1]-5,ys[r+1]-3),cell,fontsize=10)
    text(2,425,"Error codes apply only to the ML-100 model.")
    text(3,125,"Clear a paper jam using the rear access cover.")
    text(3,165,"WARNING: Turn off and unplug the printer before opening the rear cover.")
    p=doc[3]
    p.draw_rect((35,143,560,178),color=(0.7,0.25,0.1),width=1.5)
    text(3,220,"1. Press Power to turn off the printer and unplug the adapter.\n2. Open the rear access cover.\n3. Pull jammed paper slowly in the direction of paper travel.\n4. Close the rear cover until it clicks.\n5. Reconnect power and press Resume.")
    text(3,350,"CAUTION: Do not touch the rollers. Do not use sharp tools to remove paper.")
    text(3,395,"If E04 remains after these steps, contact the manufacturer support team.")
    text(4,125,"If printing is faint, run a cleaning cycle by holding Resume for five seconds.")
    text(4,165,"NOTE: Allow the cleaning cycle to finish before turning off the printer.")
    text(4,215,"If paper feeds crookedly, align the paper guides against the stack.")
    text(4,255,"If the printer does not power on, check the supplied adapter connection.")
    text(4,295,"If a USB job does not print, reconnect the USB cable and retry the job.")
    text(4,335,"For repeated blank pages, reseat the cartridge with power off.")
    text(4,375,"For an unresolved fault, contact the manufacturer support team.")
    text(5,125,"Supported paper: A4 plain paper, 80 gsm. Tray capacity: 50 sheets.")
    text(5,165,"Connectivity: USB-B only. Operating temperature: 10 to 35 degrees C.")
    text(5,205,"Dimensions: 300 x 200 x 150 mm. Mass: 2.4 kg.")
    doc.set_toc([[1,title,i+1] for i,title in enumerate(SECTIONS)])
    doc.save(path)
    doc.close()
    return path
