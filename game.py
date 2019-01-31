def start_game():
    game=[[0,0,0],[0,0,0],[0,0,0]]
    return game

def display_board(game):
    d={0:'_', 1:'x', 2:'o'}
    print(' ___')
    for row in game:
        new_row = [d[i] for i in row]
        print("|"+"|".join(new_row)+"|")
        
def check_winner(game):
    #check row
    for i in range(3):
        if game[i][0]==game[i][1]==game[i][2] !=0:
            return game[i][0]
        #check column
    for i in range(3):
        if game[0][i]==game[1][i]==game[2][i] !=0:
            return game[0][i] 
        if game[0][0]==game[1][1]==game[2][2] !=0:
            return game[i][i]
        if game[2][0]==game[1][i]==game[0][2] !=0:
            return game[1][1]
    return(0)
def switch_player(player):
    return 3-player
def isempty(game):
    for row in game:
        if 0 in row:
            return True
        
    return False
#isempty([1,1,1],)
def main():
    game = start_game()
    player = 1
    player = 0
    while winner==0 and isempty(game):
        print("player {}'s turn".format(player))
        display_board(game)
        x=int(input("input row number (start with 1)"))-1
        y=int(input("input column number (start with 1)"))-1
        game[x][y] = player
        print(game)
        display_board(game)
        winner = check_winner(game)
        if winner:
            break
        else:
            player = switch_player(player)
        if winner:
            print(")
        if check_winner(game):
            print("game over! player {} wins".format(check_winner(game)))
            
        