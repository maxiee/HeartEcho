import 'package:app/models/training_session.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:app/api/api.dart';
import 'package:app/providers/global_training_session_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

class LaunchPage extends StatefulWidget {
  final VoidCallback onSessionStart;

  const LaunchPage({super.key, required this.onSessionStart});

  @override
  State createState() => _LaunchPageState();
}

class _LaunchPageState extends State<LaunchPage> {
  TrainingSession? selectedSession;
  List<TrainingSession> savedSessions = [];
  bool isLoading = false;
  final TextEditingController _serverAddressController =
      TextEditingController();

  @override
  void initState() {
    super.initState();
    _loadSavedSessions();
    _loadServerAddress();
  }

  Future<void> _loadServerAddress() async {
    final prefs = await SharedPreferences.getInstance();
    final savedAddress =
        prefs.getString('server_address') ?? 'http://127.0.0.1:1127';
    setState(() {
      _serverAddressController.text = savedAddress;
    });
    API.baseUrl = savedAddress;
  }

  Future<void> _saveServerAddress(String address) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('server_address', address);
    API.baseUrl = address;
    _loadSavedSessions();
  }

  Future<void> _loadSavedSessions() async {
    setState(() {
      isLoading = true;
    });
    try {
      final sessions = await API.listSessions();
      setState(() {
        savedSessions = sessions;
        isLoading = false;
      });
    } catch (e) {
      if (mounted) {
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
      final sessionName = 'Session_${DateTime.now().millisecondsSinceEpoch}';
      const baseModel = 'Qwen/Qwen2-1.5B-Instruct';
      final TrainingSession session =
          await API.createNewTrainingSession(sessionName, baseModel);
      if (mounted) {
        final globalSessionProvider =
            Provider.of<GlobalTrainingSessionProvider>(context, listen: false);
        globalSessionProvider.startSession(session);
        widget.onSessionStart();
      }
    } catch (e) {
      if (mounted) {
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
      final TrainingSession loadedSession =
          await API.loadTrainingSession(selectedSession!.id);
      if (context.mounted) {
        final globalSessionProvider =
            // ignore: use_build_context_synchronously
            Provider.of<GlobalTrainingSessionProvider>(context, listen: false);
        globalSessionProvider.startSession(loadedSession);
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
                  const SizedBox(height: 24),
                  TextField(
                    controller: _serverAddressController,
                    decoration: const InputDecoration(
                      labelText: 'Server Address',
                      hintText: 'http://127.0.0.1:1127',
                      border: OutlineInputBorder(),
                    ),
                    onChanged: (value) => _saveServerAddress(value),
                  ),
                  const SizedBox(height: 24),
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
                  DropdownButton<TrainingSession>(
                    value: selectedSession,
                    hint: const Text('Select a saved session'),
                    items: savedSessions.map((session) {
                      return DropdownMenuItem<TrainingSession>(
                        value: session,
                        child: Text(session.name),
                      );
                    }).toList(),
                    onChanged: isLoading
                        ? null
                        : (TrainingSession? newValue) {
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
