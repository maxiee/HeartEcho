import 'package:app/models/corpus.dart';
import 'package:flutter/material.dart';

class LearnChatPage extends StatefulWidget {
  const LearnChatPage({super.key, this.entry});

  final CorpusEntry? entry;

  @override
  State<LearnChatPage> createState() => _LearnChatPageState();
}

class _LearnChatPageState extends State<LearnChatPage> {
  List<Map<String, String>> messages = [];
  TextEditingController contentController = TextEditingController();
  String currentRole = 'user';

  @override
  void initState() {
    super.initState();
    // promptController.text = systemPromot;
  }

  @override
  void dispose() {
    super.dispose();
    contentController.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Create Chat Corpus'),
        actions: [
          IconButton(
            icon: const Icon(Icons.save),
            onPressed: () => _saveCorpus(context),
          )
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              itemCount: messages.length,
              itemBuilder: (context, index) {
                final message = messages[index];
                return ListTile(
                  leading: Text(message['role']!),
                  title: Text(message['content']!),
                );
              },
            ),
          ),
          Row(
            children: [
              DropdownButton<String>(
                value: currentRole,
                items: ['user', 'assistant', 'system']
                    .map((role) =>
                        DropdownMenuItem(value: role, child: Text(role)))
                    .toList(),
                onChanged: (value) => setState(() => currentRole = value!),
              ),
              Expanded(
                child: TextField(
                  controller: contentController,
                  decoration: const InputDecoration(hintText: 'Enter message'),
                ),
              ),
              IconButton(
                icon: const Icon(Icons.send),
                onPressed: () {
                  setState(() {
                    messages.add({
                      'role': currentRole,
                      'content': contentController.text,
                    });
                    contentController.clear();
                  });
                },
              ),
            ],
          ),
        ],
      ),
    );
  }

  void _saveCorpus(BuildContext context) {
    // Implement API call to save the corpus
    // Use Provider.of<BatchProvider>(context, listen: false).addChatSession(messages);
  }
}
