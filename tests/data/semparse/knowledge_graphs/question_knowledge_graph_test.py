# pylint: disable=no-self-use,invalid-name,protected-access,too-many-public-methods
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.semparse.knowledge_graphs import QuestionKnowledgeGraph
from allennlp.data import Token


class TestQuestionKnowledgeGraph(AllenNlpTestCase):
    def test_read_handles_simple_cases(self):
        question = [Token(x) for x in ['Mary', 'bought', '10', 'tickets', 'for', '$150' ]]
        graph = QuestionKnowledgeGraph.read(question)

        assert graph.entities == ['10', '150']
        assert graph.neighbors['10'] == []
        assert graph.neighbors['150'] == []
        assert graph.entity_text['10'] == '10'
        assert graph.entity_text['150'] == '$150'

