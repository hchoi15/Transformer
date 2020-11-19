import xml.etree.ElementTree as ET

doc = ET.parse("./dataset/de-en/IWSLT16.TED.tst2010.de-en.de.xml")
#parse(): XML 섹션을 엘리먼트 트리로 구문 분석

root = doc.getroot()
#getroot(): 이 트리의 루트 엘리먼트를 반환

with open('./dataset/de-en/test.de',"w") as f:
    for seg in root.iter("seg"):
        #iter(): 루트 엘리먼트에 대한 트리 이터레이터를 만들고 반환 (tag=찾을 태그입니다, 기본값은 모든 엘리먼트를 반환)
        f.write(seg.text+"\n")

doc = ET.parse("./dataset/de-en/IWSLT16.TED.tst2010.de-en.en.xml")

root = doc.getroot()

with open('./dataset/de-en/test.en',"w") as f:
    for seg in root.iter("seg"):
        f.write(seg.text +"\n")