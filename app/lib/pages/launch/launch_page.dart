import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:app/api/api.dart';
import 'package:app/providers/global_training_session_provider.dart';

class LaunchPage extends StatefulWidget {
  final VoidCallback onSessionStart;

  const LaunchPage({super.key, required this.onSessionStart});

  @override
  State createState() => _LaunchPageState();
}

class _LaunchPageState extends State<LaunchPage> {
  String? selectedSession;
  List<String> savedSessions = [];
  bool isLoading = false;

  @override
  void initState() {
    super.initState();
    _loadSavedSessions();
  }

  Future<void> _loadSavedSessions() async {
    setState(() {
      isLoading = true;
    });
    try {
      final sessions = await API.getSavedModels();
      setState(() {
        savedSessions = sessions;
        isLoading = false;
      });
    } catch (e) {
      if (context.mounted) {
        // ignore: use_build_context_synchronously
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error loading saved sessions: $e')),
        );
      }
      setState(() {
        isLoading = false;
      });
    }
  }

  Future<void> _startNewGame() async {
    setState(() {
      isLoading = true;
    });
    try {
      final sessionName = await API.createNewTrainingSession();
      if (context.mounted) {
        final globalSessionProvider =
            // ignore: use_build_context_synchronously
            Provider.of<GlobalTrainingSessionProvider>(context, listen: false);
        globalSessionProvider.startSession(sessionName);
        widget.onSessionStart();
      }
    } catch (e) {
      if (context.mounted) {
        // ignore: use_build_context_synchronously
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error starting new session: $e')),
        );
      }
      setState(() {
        isLoading = false;
      });
    }
  }

  Future<void> _loadGame() async {
    if (selectedSession == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please select a saved session')),
      );
      return;
    }

    setState(() {
      isLoading = true;
    });
    try {
      await API.loadExistingModel(selectedSession!);
      if (context.mounted) {
        final globalSessionProvider =
            // ignore: use_build_context_synchronously
            Provider.of<GlobalTrainingSessionProvider>(context, listen: false);
        globalSessionProvider.startSession(selectedSession!);
        widget.onSessionStart();
      }
    } catch (e) {
      if (context.mounted) {
        // ignore: use_build_context_synchronously
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error loading session: $e')),
        );
      }
      setState(() {
        isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Colors.purple[900]!, Colors.blue[900]!],
          ),
        ),
        child: Center(
          child: Card(
            elevation: 8,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            color: Colors.white.withOpacity(0.9),
            child: Padding(
              padding: const EdgeInsets.all(32.0),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    'HeartEcho',
                    style: Theme.of(context).textTheme.displayLarge?.copyWith(
                          color: Colors.purple[900],
                          fontWeight: FontWeight.bold,
                        ),
                  ),
                  const SizedBox(height: 24),
                  Text(
                    'Your Personal AI Companion',
                    style: Theme.of(context).textTheme.displaySmall,
                  ),
                  const SizedBox(height: 48),
                  ElevatedButton(
                    onPressed: isLoading ? null : _startNewGame,
                    style: ElevatedButton.styleFrom(
                      // primary: Colors.purple[700],
                      padding: const EdgeInsets.symmetric(
                          horizontal: 32, vertical: 16),
                    ),
                    child: const Text('New Game'),
                  ),
                  const SizedBox(height: 24),
                  DropdownButton<String>(
                    value: selectedSession,
                    hint: const Text('Select a saved session'),
                    items: savedSessions.map((String session) {
                      return DropdownMenuItem<String>(
                        value: session,
                        child: Text(session),
                      );
                    }).toList(),
                    onChanged: isLoading
                        ? null
                        : (String? newValue) {
                            setState(() {
                              selectedSession = newValue;
                            });
                          },
                  ),
                  const SizedBox(height: 16),
                  ElevatedButton(
                    onPressed: isLoading ? null : _loadGame,
                    style: ElevatedButton.styleFrom(
                      // primary: Colors.blue[700],
                      padding: const EdgeInsets.symmetric(
                          horizontal: 32, vertical: 16),
                    ),
                    child: const Text('Load Game'),
                  ),
                  if (isLoading)
                    const Padding(
                      padding: EdgeInsets.all(16.0),
                      child: CircularProgressIndicator(),
                    ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
