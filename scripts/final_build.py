"""Final build: all 3 slides modified, bullet list properly formatted."""
import zipfile, re, xml.etree.ElementTree as ET, copy

SRC = "D:/英语选修/python_project_preaentation_final_v2.pptx"
OUT = "D:/英语选修/python_project_preaentation_final_final.pptx"

NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"

print("Building final version...")
with zipfile.ZipFile(SRC, 'r') as zin:
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            m = re.match(r'ppt/slides/slide(\d+)\.xml', item.filename)
            if m:
                sn = int(m.group(1))
                if sn in [18, 20]:
                    root = ET.fromstring(data)
                    for sp in root.findall(f'.//{{{NS_P}}}sp'):
                        name = sp.find(f'.//{{{NS_P}}}cNvPr').get('name', '')
                        t_els_all = list(sp.iter(f'{{{NS_A}}}t'))
                        
                        # Helper: replace text in shape
                        def rep(old, new):
                            full = ''.join(t.text or '' for t in t_els_all)
                            if old in full and t_els_all:
                                t_els_all[0].text = full.replace(old, new)
                                for t in t_els_all[1:]: t.text = ''
                                return True
                            return False
                        
                        # === SLIDE 18 ===
                        if sn == 18:
                            if name == '文本框 14':
                                rep('关键事实召回率几乎翻倍',
                                    '53题评估：混合检索在98%题目上优于纯LLM，平均提升+0.51')
                            elif name == '文本框 15':
                                rep('增加了\u201c严格依据参考信息回答，不足则拒答\u201d的强约束,具备刚性准确度',
                                    '已增加智能路由：simple题放松约束、rule题严格依据、检索低分回退纯LLM')
                            elif name == '文本框 5':
                                rep('接入 RAG 后',
                                    '53题评估：纯LLM平均0.20 \u2192 向量RAG 0.41 \u2192 混合RAG 0.71（+0.51）。'
                                    '规则细节题尤为突出：证件列表从0.02\u21920.75、行李保管从0.06\u21920.87。'
                                    '混合检索让轻量模型在铁路领域知识上获得质的飞跃。')
                        
                        # === SLIDE 20 ===
                        elif sn == 20:
                            if name == '文本框 26':
                                # Handle bullet list: each ①②③④ is a separate <a:p>
                                for p in sp.findall(f'.//{{{NS_A}}}p'):
                                    t_els = list(p.findall(f'.//{{{NS_A}}}t'))
                                    text = ''.join(t.text or '' for t in t_els)
                                    if '\u2460' in text and '混合检索' in text and t_els:
                                        t_els[0].text = (
                                            '\u2460 \u2705混合检索：已实现 FAISS向量 + BM25字符级n-gram + '
                                            'RRF融合双路检索。53题评估：98%题目优于纯LLM，平均提升+0.51'
                                        )
                                        for t in t_els[1:]: t.text = ''
                            
                            elif name == '文本框 15':
                                # Fix ②, and add properly separated ⑤
                                for p in sp.findall(f'.//{{{NS_A}}}p'):
                                    t_els = list(p.findall(f'.//{{{NS_A}}}t'))
                                    text = ''.join(t.text or '' for t in t_els)
                                    
                                    if '\u2461' in text and '检索依赖' in text and t_els:
                                        t_els[0].text = (
                                            '\u2461 已升级为双路检索，但仍有检索失败案例'
                                            '（如证件列表题向量仅0.02），需改进chunking与query改写'
                                        )
                                        for t in t_els[1:]: t.text = ''
                                    
                                    elif '\u2463' in text and 'Ollama' in text:
                                        # Append ⑤ to the ④ paragraph, properly spaced
                                        if t_els:
                                            t_els[0].text = text + '\n\u2464 当前评估基于轻量模型(qwen2.5:0.5b)，大模型下混合检索获益幅度可能收窄，需进一步验证'
                                            for t in t_els[1:]: t.text = ''
                    
                    data = ET.tostring(root, encoding='utf-8', xml_declaration=True)
            zout.writestr(item, data)

# Final verification
print("\n=== Final Check ===")
with zipfile.ZipFile(OUT, 'r') as z:
    for sf in ['ppt/slides/slide18.xml', 'ppt/slides/slide20.xml']:
        root = ET.fromstring(z.read(sf))
        sn = re.search(r'slide(\d+)', sf).group(1)
        print(f'--- Slide {sn} ---')
        for sp in root.findall(f'.//{{{NS_P}}}sp'):
            name = sp.find(f'.//{{{NS_P}}}cNvPr').get('name', '')
            if name in ['文本框 14', '文本框 15', '文本框 5', '文本框 26']:
                print(f'  [{name}]')
                for p in sp.findall(f'{{{NS_A}}}p'):
                    t = ''.join(t_e.text or '' for t_e in p.findall(f'.//{{{NS_A}}}t'))
                    if t.strip(): print(f'    {t.strip()[:160]}')
        print()

print(f'Done: {OUT}')
