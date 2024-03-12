import 'package:app/api/api.dart';
import 'package:code_text_field/code_text_field.dart';
import 'package:flutter/material.dart';
import 'package:flutter/widgets.dart';

class LearnChatPage extends StatefulWidget {
  const LearnChatPage({super.key});

  @override
  State<LearnChatPage> createState() => _LearnChatPageState();
}

class _LearnChatPageState extends State<LearnChatPage> {
  String systemPromot = '你是一个有用的助手。';
  TextEditingController promptController = TextEditingController();
  TextEditingController questionController = TextEditingController();
  CodeController answerController = CodeController();

  @override
  void initState() {
    super.initState();
    promptController.text = systemPromot;
  }

  @override
  void dispose() {
    super.dispose();
    promptController.dispose();
    questionController.dispose();
    answerController.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        body: Column(
          children: [
            Text("系统提示："),
            TextField(
              controller: promptController,
            ),
            Text('问题：'),
            TextField(
              controller: questionController,
            ),
            Text('回答：'),
            Expanded(
              child: CodeField(
                controller: answerController,
                expands: true,
                wrap: true,
                textStyle: TextStyle(fontFamily: 'SourceCode')),
            ),
          ],
        ),
        floatingActionButton: FloatingActionButton(
          onPressed: () {
            if (promptController.text.isEmpty ||
                questionController.text.isEmpty ||
                answerController.text.isEmpty) return;
            final data = [
              {
                "role": "system",
                "content": promptController.text,
              },
              {
                "role": "user",
                "content": questionController.text,
              },
              {
                "role": "assistant",
                "content": answerController.text,
              }
            ];
            print(data);
            // chatml
            API.learnChat(data).then((response) {
              setState(() {
                questionController.clear();
                answerController.clear();
              });
            });
          },
          child: Text('学', style: TextStyle(fontWeight: FontWeight.bold)),
        ));
  }
}
