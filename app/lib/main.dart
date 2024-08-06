import 'package:app/components/global_titlebar.dart';
import 'package:app/global_provider.dart';
import 'package:app/pages/chat/chat_page.dart';
import 'package:app/pages/corpus/corpus_management_page.dart';
import 'package:app/pages/train/train_page.dart';
import 'package:app/providers/batch_provider.dart';
import 'package:app/providers/global_training_session_provider.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

void main() {
  runApp(MultiProvider(providers: [
    ChangeNotifierProvider(create: (context) => GlobalProvider()),
    ChangeNotifierProvider(create: (context) => BatchProvider()),
    ChangeNotifierProvider(
        create: (context) => GlobalTrainingSessionProvider()),
  ], child: const MyApp()));
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
  @override
  Widget build(BuildContext context) {
    return Scaffold(
        body: Column(
      children: [
        const GlobalTitlebar(),
        Expanded(child: getPage(context)),
      ],
    ));
  }

  getPage(BuildContext context) {
    final globalProvider = Provider.of<GlobalProvider>(context);
    final currentMode = globalProvider.mode;
    if (currentMode == Mode.Chat) return const ChatPage();
    if (currentMode == Mode.Corpus) return const CorpusManagementPage();
    if (currentMode == Mode.Train) return const TrainPage();
    return const Placeholder();
  }
}
