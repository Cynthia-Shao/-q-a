"""Rebuild v4: slides 11+18+20 all modified, with tuned messaging."""
import zipfile, re
import xml.etree.ElementTree as ET

SRC = "D:/英语选修/python_project_preaentation_final_v2.pptx"  # only slide 11 done
OUT = "D:/英语选修/python_project_preaentation_final_v4.pptx"

NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"

def do_replace(sp, name, old_substr, new_text):
    t_els = list(sp.iter(f'{{{NS_A}}}t'))
    text = ''.join(t.text or '' for t in t_els)
    if old_substr in text and t_els:
        t_els[0].text = new_text
        for t in t_els[1:]: t.text = ''
        return True
    return False

def do_append_bullet(sp, name, search_char, append_str):
    """Find paragraph containing search_char and append text to it."""
    for p in sp.findall(f'.//{{{NS_A}}}p'):
        t_els = list(p.findall(f'.//{{{NS_A}}}t'))
        text = ''.join(t.text or '' for t in t_els)
        if search_char in text and t_els:
            t_els[0].text = text + append_str
            for t in t_els[1:]: t.text = ''
            return True
    return False

def do_replace_para(sp, name, search_char, new_text):
    """Replace paragraph containing search_char."""
    for p in sp.findall(f'.//{{{NS_A}}}p'):
        t_els = list(p.findall(f'.//{{{NS_A}}}t'))
        text = ''.join(t.text or '' for t in t_els)
        if search_char in text and t_els:
            t_els[0].text = new_text
            for t in t_els[1:]: t.text = ''
            return True
    return False

print("Building v4...")
with zipfile.ZipFile(SRC, 'r') as zin:
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            
            m = re.match(r'ppt/slides/slide(\d+)\.xml', item.filename)
            if m:
                sn = int(m.group(1))
                if sn == 11:
                    # Already done in v2, skip
                    pass
                elif sn in [18, 20]:
                    root = ET.fromstring(data)
                    for sp in root.findall(f'.//{{{NS_P}}}sp'):
                        name = sp.find(f'.//{{{NS_P}}}cNvPr').get('name', '')
                        
                        if sn == 18:
                            if name == '文本框 14':
                                if do_replace(sp, name, '关键事实召回率几乎翻倍',
                                    '53题评估：混合检索在98%题目上优于纯LLM，平均提升+0.51'):
                                    print(f'  Slide 18 [{name}] ok')
                            elif name == '文本框 15':
                                if do_replace(sp, name, '严格依据参考信息回答，不足则拒答',
                                    '已增加智能路由：simple题放松约束、rule题严格依据、检索低分回退纯LLM'):
                                    print(f'  Slide 18 [{name}] ok')
                            elif name == '文本框 5':
                                if do_replace(sp, name, '语义提升值',
                                    '53题评估：纯LLM平均0.20 \u2192 向量RAG 0.41 \u2192 混合RAG 0.71（+0.51）。'
                                    '规则细节题尤为突出：证件列表从0.02\u21920.75、行李保管从0.06\u21920.87。'
                                    '混合检索让轻量模型在铁路领域知识上获得质的飞跃。'):
                                    print(f'  Slide 18 [{name}] ok')
                        
                        elif sn == 20:
                            if name == '文本框 26':
                                if do_replace_para(sp, name, '\u2460 混合检索',
                                    '\u2460 \u2705混合检索：已实现 FAISS向量 + BM25字符级n-gram + RRF融合'
                                    '双路检索。53题评估：98%题目优于纯LLM，平均提升+0.51'):
                                    print(f'  Slide 20 [{name}] \u2460 ok')
                            elif name == '文本框 15':
                                if do_replace_para(sp, name, '\u2461 检索依赖',
                                    '\u2461 已升级为双路检索，但仍有检索失败案例'
                                    '（如证件列表题向量仅0.02），需改进chunking与query改写'):
                                    pass  # already done, just showing
                                if do_append_bullet(sp, name, '\u2463',
                                    '\u2464 当前评估基于轻量模型(qwen2.5:0.5b)，大模型下混合检索获益幅度可能收窄，需进一步验证'):
                                    print(f'  Slide 20 [{name}] appended \u2464')
                    
                    data = ET.tostring(root, encoding='utf-8', xml_declaration=True)
            
            zout.writestr(item, data)

# Verify
print("\n=== Verification ===")
with zipfile.ZipFile(OUT, 'r') as z:
    for sf in ['ppt/slides/slide18.xml', 'ppt/slides/slide20.xml']:
        root = ET.fromstring(z.read(sf))
        for sp in root.findall(f'.//{{{NS_P}}}sp'):
            name = sp.find(f'.//{{{NS_P}}}cNvPr').get('name', '')
            targets = ['文本框 14', '文本框 15', '文本框 5', '文本框 26']
            if name in targets:
                print(f'  [{name}]')
                for p in sp.findall(f'{{{NS_A}}}p'):
                    t = ''.join(t_e.text or '' for t_e in p.findall(f'.//{{{NS_A}}}t'))
                    if t.strip(): print(f'    {t.strip()[:160]}')
        print()

print(f'Done: {OUT}')
