import 'package:app/api/api.dart';
import 'package:app/components/global_titlebar.dart';
import 'package:app/global_provider.dart';
import 'package:app/pages/chat/chat_page.dart';
import 'package:app/pages/corpus/corpus_management_page.dart';
import 'package:app/pages/launch/launch_page.dart';
import 'package:app/providers/new_corpus_entries_provider.dart';
import 'package:app/pages/train/train_page.dart';
import 'package:app/providers/batch_provider.dart';
import 'package:app/providers/global_training_session_provider.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:window_manager/window_manager.dart';

void main() {
  runApp(MultiProvider(providers: [
    ChangeNotifierProvider(create: (context) => GlobalProvider()),
    ChangeNotifierProvider(create: (context) => BatchProvider()),
    ChangeNotifierProvider(
        create: (context) => GlobalTrainingSessionProvider()),
    ChangeNotifierProvider(create: (context) => NewCorpusEntriesProvider()),
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
      home: const MyHomePage(title: 'HeartEcho'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});
  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> with WindowListener {
  bool _sessionStarted = false;

  @override
  void initState() {
    super.initState();
    windowManager.addListener(this);
    _init();
  }

  @override
  void dispose() {
    windowManager.removeListener(this);
    super.dispose();
  }

  void _init() async {
    // Add this line to override the default close handler
    await windowManager.setPreventClose(true);
    setState(() {});
  }

  void _onSessionStart() {
    setState(() {
      _sessionStarted = true;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (!_sessionStarted) {
      return LaunchPage(onSessionStart: _onSessionStart);
    }

    return Scaffold(
      body: Column(
        children: [
          const GlobalTitlebar(),
          Expanded(child: getPage(context)),
        ],
      ),
    );
  }

  getPage(BuildContext context) {
    final globalProvider = Provider.of<GlobalProvider>(context);
    final currentMode = globalProvider.mode;
    if (currentMode == Mode.Chat) return const ChatPage();
    if (currentMode == Mode.Corpus) return const CorpusManagementPage();
    if (currentMode == Mode.Train) return const TrainPage();
    return const Placeholder();
  }

  @override
  void onWindowClose() async {
    bool _isPreventClose = await windowManager.isPreventClose();
    if (_isPreventClose) {
      // Show a dialog to inform the user that the app is saving and closing
      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (_) {
          return AlertDialog(
            title: Text('Saving and closing...'),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                CircularProgressIndicator(),
                SizedBox(height: 16),
                Text('Please wait while we save your model.'),
              ],
            ),
          );
        },
      );

      try {
        // Get necessary providers
        final globalSessionProvider =
            Provider.of<GlobalTrainingSessionProvider>(context, listen: false);

        // Save current session
        final newSession = await API.saveCurrentSession();

        // Update the initial tokens trained
        globalSessionProvider
            .updateInitialTokensTrained(newSession.tokensTrained);

        // Show success message (optional, since we're closing the app anyway)
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Session saved successfully')),
        );

        // Close the app after saving
        await windowManager.destroy();
      } catch (e) {
        // If there's an error, show it to the user
        Navigator.of(context).pop(); // Close the "Saving and closing" dialog
        showDialog(
          context: context,
          builder: (_) {
            return AlertDialog(
              title: Text('Error'),
              content: Text('Failed to save the session: $e'),
              actions: [
                TextButton(
                  child: Text('OK'),
                  onPressed: () => Navigator.of(context).pop(),
                ),
              ],
            );
          },
        );
      }
    }
  }
}
