from stanza.utils.conll import CoNLL
import stanza
stanza.download('en')
nlp_stanza = stanza.Pipeline('en')

def sent_to_conll10(sentences: list) -> str:
    result = ''
    for sent in sentences:
        st = nlp_stanza(sent)
        result += '\n'.join(['\t'.join(s) for s in CoNLL.convert_dict(st.to_dict())[-1]])
        result += '\n'
    return result
