import 'package:app/api/api.dart';
import 'package:app/models/corpus.dart';
import 'package:flutter/material.dart';

class LearnChatPage extends StatefulWidget {
  const LearnChatPage({super.key, required this.corpus, this.entry});

  final Corpus corpus;
  final CorpusEntry? entry;

  @override
  State<LearnChatPage> createState() => _LearnChatPageState();
}

class _LearnChatPageState extends State<LearnChatPage> {
  List<Message> messages = [];
  TextEditingController contentController = TextEditingController();
  String currentRole = 'user';

  @override
  void initState() {
    super.initState();
    if (widget.entry != null) {
      messages = widget.entry!.messages ?? [];
    }
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
        title:
            Text(widget.entry == null ? 'Add Chat Entry' : 'Edit Chat Entry'),
        actions: [
          IconButton(
            icon: const Icon(Icons.save),
            onPressed: () => _saveEntry(context),
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
                  leading: Text(message.role),
                  title: Text(message.content),
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
                    messages.add(Message(
                        role: currentRole, content: contentController.text));
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

  void _saveEntry(BuildContext context) async {
    try {
      final Map<String, dynamic> data = {
        'corpus_id': widget.corpus.id,
        'entry_type': 'chat',
        'messages': messages,
      };

      if (widget.entry == null) {
        await API.createCorpusEntry(data);
      } else {
        // Assuming you have an API method to update an entry
        await API.updateCorpusEntry(widget.entry!.id, data);
      }

      if (context.mounted) {
        Navigator.pop(context, true); // Return true to indicate success
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error saving entry: $e')),
        );
      }
    }
  }
}
