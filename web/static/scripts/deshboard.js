$( function() {
  var $winHeight = $( window ).height()
  $( 'indector.container' ).height( $winHeight );
});

$(document).ready(function(){
    $('#test_alexNet,#test_TEST').on('click',function(){
        var _this = this;
        setTimeout(function(){
            if($(_this).parents('div:first').next().css('display')!='block'){
                $(_this).parents('div:first').next().css('display','block');
            }
        },3500);

    });

    $('[id^="continue_"]').on('click', function() {
        var networkName = $(this).parents('div:first').attr('net-data');
        $.ajax({
            url: "/train/continue/",
            type: "POST",
            contentType: 'application/json; charset=utf-8',
            data: networkName,
            success: function(result) {
                alert(result);
            }
        });
    });

    $('[id^="stop_"]').on('click', function() {
        var networkName = $(this).parents('div:first').attr('net-data');
        $.ajax({
            url: "train/stop/",
            type: "POST",
            contentType: 'application/json; charset=utf-8',
            data: networkName,
            success: function(result) {
                alert(result);
            }
        });
    });
});