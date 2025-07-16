import unittest

class TestChat(unittest.TestCase):
    def test_chat_with_many_concepts_limit(self):
        """Test chat with many concepts and limit."""
        # Create many concepts
        concepts = [
            Concept(
                name=f"Concept{i}",
                description=f"Description{i}",
                category="test",
                source="test"
            )
            for i in range(100)
        ]
        
        # Add concepts to knowledge base
        for concept in concepts:
            self.chat.add_concept(concept)
        
        # Test chat with limit
        response = self.chat.chat("Tell me about the concepts", limit=5)
        
        # Check response
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        
        # Check that only 5 concepts were used
        self.assertEqual(len(self.chat.knowledge_base.concepts), 100)
        self.assertEqual(len(self.chat.knowledge_base.get_relevant_concepts("test", limit=5)), 5) 