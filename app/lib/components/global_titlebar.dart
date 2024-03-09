import 'package:app/global_provider.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

class GlobalTitlebar extends StatelessWidget {
  const GlobalTitlebar({super.key});

  @override
  Widget build(BuildContext context) {
    final globalProvider = Provider.of<GlobalProvider>(context);
    final currentMode = globalProvider.mode;
    return SizedBox(
      height: 50,
      child: Stack(
        children: [
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Text(
              "HeartEcho",
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
          Align(
            alignment: Alignment.center,
            child: Row(mainAxisSize: MainAxisSize.min, children: [
              TextButton(
                  onPressed: () {
                    if (currentMode == Mode.Chat) return;
                    globalProvider.changeMode(Mode.Chat);
                  },
                  child: Text('聊天',
                      style: TextStyle(
                          color: currentMode == Mode.Chat
                              ? Colors.blue
                              : Colors.black))),
              TextButton(
                  onPressed: () {
                    if (currentMode == Mode.LearnKnowledge) return;
                    globalProvider.changeMode(Mode.LearnKnowledge);
                  },
                  child: Text('学知识',
                      style: TextStyle(
                          color: currentMode == Mode.LearnKnowledge
                              ? Colors.blue
                              : Colors.black))),
              TextButton(
                  onPressed: () {
                    if (currentMode == Mode.LearnChat) return;
                    globalProvider.changeMode(Mode.LearnChat);
                  },
                  child: Text('学聊天',
                      style: TextStyle(
                          color: currentMode == Mode.LearnChat
                              ? Colors.blue
                              : Colors.black))),
            ]),
          ),
          Align(
              alignment: Alignment.bottomRight,
              child: Container(
                height: 1,
                color: Colors.grey,
              ))
        ],
      ),
    );
  }
}
