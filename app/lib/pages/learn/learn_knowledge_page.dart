import 'package:app/api/api.dart';
import 'package:app/global_provider.dart';
import 'package:app/providers/batch_provider.dart';
import 'package:code_text_field/code_text_field.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

class LearnKnowledgePage extends StatefulWidget {
  const LearnKnowledgePage({super.key});

  @override
  State<LearnKnowledgePage> createState() => _LearnKnowledgePageState();
}

class _LearnKnowledgePageState extends State<LearnKnowledgePage> {
  late CodeController _codeController;

  @override
  void initState() {
    super.initState();
    _codeController = CodeController(
      text: '',
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
        body: CodeField(
            controller: _codeController,
            expands: true,
            wrap: true,
            textStyle: TextStyle(fontFamily: 'SourceCode')),
        floatingActionButton: FloatingActionButton(
          onPressed: () {
            if (_codeController.text.isEmpty) return;
            Provider.of<BatchProvider>(context, listen: false)
                .addKnowledge(_codeController.text);
          },
          child: Text('学', style: TextStyle(fontWeight: FontWeight.bold)),
        ));
  }
}
