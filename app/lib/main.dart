import 'package:app/components/global_titlebar.dart';
import 'package:app/global_provider.dart';
import 'package:app/pages/chat/chat_page.dart';
import 'package:app/pages/learn/learn_chat_page.dart';
import 'package:app/pages/learn/learn_knowledge_page.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

void main() {
  runApp(ChangeNotifierProvider(
      create: (context) => GlobalProvider(), child: const MyApp()));
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'HeartEcho',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      debugShowCheckedModeBanner: false,
      home: const MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});
  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  int _counter = 0;

  void _incrementCounter() {
    setState(() {
      _counter++;
    });
  }

  @override
  Widget build(BuildContext context) {
    final globalProvider = Provider.of<GlobalProvider>(context);
    final currentMode = globalProvider.mode;
    return Scaffold(
        body: Column(
      children: [
        GlobalTitlebar(),
        Expanded(child: getPage(context)),
      ],
    ));
  }

  getPage(BuildContext context) {
    final globalProvider = Provider.of<GlobalProvider>(context);
    final currentMode = globalProvider.mode;
    if (currentMode == Mode.Chat) return ChatPage();
    if (currentMode == Mode.LearnKnowledge) return LearnKnowledgePage();
    if (currentMode == Mode.LearnChat) return LearnChatPage();
    return Placeholder();
  }
}
