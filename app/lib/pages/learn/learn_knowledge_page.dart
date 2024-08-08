import 'package:app/api/api.dart';
import 'package:app/models/corpus.dart';
import 'package:code_text_field/code_text_field.dart';
import 'package:flutter/material.dart';

class LearnKnowledgePage extends StatefulWidget {
  final Corpus corpus;
  final CorpusEntry? entry;

  const LearnKnowledgePage({super.key, required this.corpus, this.entry});

  @override
  State<LearnKnowledgePage> createState() => _LearnKnowledgePageState();
}

class _LearnKnowledgePageState extends State<LearnKnowledgePage> {
  late CodeController _codeController;

  @override
  void initState() {
    super.initState();
    _codeController = CodeController(
      text: widget.entry?.content ?? '',
    );
  }

  @override
  void dispose() {
    super.dispose();
    _codeController.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.entry == null
            ? 'Add Knowledge Entry'
            : 'Edit Knowledge Entry'),
        actions: [
          IconButton(
            icon: const Icon(Icons.save),
            onPressed: () => _saveEntry(context),
          )
        ],
      ),
      body: CodeField(
          controller: _codeController,
          expands: true,
          wrap: true,
          textStyle: const TextStyle(fontFamily: 'SourceCode')),
    );
  }

  void _saveEntry(BuildContext context) async {
    try {
      final Map<String, dynamic> data = {
        'corpus_id': widget.corpus.id,
        'entry_type': 'knowledge',
        'content': _codeController.text,
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
