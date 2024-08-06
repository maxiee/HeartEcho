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
          const Padding(
            padding: EdgeInsets.all(8.0),
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
                  child: Text('聊天室',
                      style: TextStyle(
                          color: currentMode == Mode.Chat
                              ? Colors.blue
                              : Colors.black))),
              TextButton(
                  onPressed: () {
                    if (currentMode == Mode.Corpus) return;
                    globalProvider.changeMode(Mode.Corpus);
                  },
                  child: Text('语料库',
                      style: TextStyle(
                          color: currentMode == Mode.Corpus
                              ? Colors.blue
                              : Colors.black))),
              TextButton(
                  onPressed: () {
                    if (currentMode == Mode.Train) return;
                    globalProvider.changeMode(Mode.Train);
                  },
                  child: Text('炼丹炉',
                      style: TextStyle(
                          color: currentMode == Mode.Train
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
